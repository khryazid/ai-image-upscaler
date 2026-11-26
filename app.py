import os
import io
import uuid
import time
import zipfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_file
from colorizer import colorize_image

try:
    import onnxruntime as ort
    has_ort = True
except ImportError:
    has_ort = False

# --- Optimizations & Device Setup ---
# Limit PyTorch to 4 physical cores to avoid Hyper-Threading bottlenecks
torch.set_num_threads(4)
# Prevent OpenCV from competing for CPU threads
cv2.setNumThreads(0)
device = torch.device('cpu')
# ------------------------------------

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# Official Real-ESRGAN model from GitHub (public, no auth needed)
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = MODEL_DIR / "RealESRGAN_x4plus.pth"
ONNX_PATH = MODEL_DIR / "RealESRGAN_x4plus.onnx"

_upscaler = None
_ort_session = None


# ---- RRDBNet Architecture (inline to avoid basicsr dependency) ----

class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(torch.nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(torch.nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = torch.nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = torch.nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = torch.nn.Conv2d(nf, nf, 3, 1, 1)
        # Upsampling
        self.conv_up1 = torch.nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # Upsample 4x (2x twice)
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ---- Download & Load ----

def download_model():
    """Download the model if not present."""
    if MODEL_PATH.exists():
        return
    print("[*] Downloading Real-ESRGAN model (~64MB)...")
    print(f"    From: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"[OK] Model downloaded ({size_mb:.1f} MB)")


def get_upscaler():
    """Lazy-load the model."""
    global _upscaler, _ort_session
    if _ort_session is not None: return _ort_session
    if _upscaler is not None: return _upscaler
    
    download_model()

    if has_ort:
        if not ONNX_PATH.exists():
            print("[*] Converting Real-ESRGAN to ONNX format for GPU acceleration. This may take a minute...")
            model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
            state_dict = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=True)
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            dummy_input = torch.randn(1, 3, 256, 256)
            torch.onnx.export(model, dummy_input, str(ONNX_PATH), 
                              opset_version=14, 
                              input_names=['input'], 
                              output_names=['output'], 
                              dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 
                                            'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
            print("[OK] ONNX conversion complete")
            del model

        print("[*] Loading ONNX session with DirectML (Intel GPU)")
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        _ort_session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        print("[OK] ONNX Model loaded and ready")
        return _ort_session

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    state_dict = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=True)
    # Handle 'params_ema' key wrapping
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    _upscaler = model
    print("[OK] Model loaded and ready")
    return _upscaler


# ---- Upscale Logic ----

def upscale_tile(model_or_session, tile):
    """Upscale a single tile. Input: HWC BGR uint8. Output: HWC BGR uint8."""
    # BGR uint8 -> RGB float32 [0,1]
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    if has_ort and hasattr(model_or_session, 'run'):
        input_data = np.transpose(tile_rgb, (2, 0, 1))[np.newaxis, ...] # NCHW
        ort_inputs = {model_or_session.get_inputs()[0].name: input_data}
        output = model_or_session.run(None, ort_inputs)[0]
        output = output[0].transpose(1, 2, 0) # HWC
        output = np.clip(output, 0, 1)
    else:
        tensor = torch.from_numpy(tile_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # NCHW
        with torch.no_grad():
            output = model_or_session(tensor)
        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    output = (output * 255.0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def upscale_image(img, scale=4, tile_size=256, tile_pad=16):
    """Upscale an image using tiled processing."""
    model = get_upscaler()
    h, w, c = img.shape

    # Small images: process directly
    if h <= tile_size and w <= tile_size:
        result = upscale_tile(model, img)
        if scale == 2:
            rh, rw = result.shape[:2]
            result = cv2.resize(result, (rw // 2, rh // 2), interpolation=cv2.INTER_AREA)
        return result

    # Tiled processing for larger images
    out_h, out_w = h * 4, w * 4
    output = np.zeros((out_h, out_w, c), dtype=np.float64)
    weights = np.zeros((out_h, out_w, c), dtype=np.float64)

    tiles_y = max(1, (h + tile_size - 1) // tile_size)
    tiles_x = max(1, (w + tile_size - 1) // tile_size)
    total_tiles = tiles_y * tiles_x
    done = 0

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y1 = ty * tile_size
            x1 = tx * tile_size
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            # Padded region
            py1 = max(0, y1 - tile_pad)
            px1 = max(0, x1 - tile_pad)
            py2 = min(h, y2 + tile_pad)
            px2 = min(w, x2 + tile_pad)

            tile = img[py1:py2, px1:px2]
            result_tile = upscale_tile(model, tile)

            # Calculate output region (non-padded)
            oy1 = (y1 - py1) * 4
            ox1 = (x1 - px1) * 4
            oy2 = oy1 + (y2 - y1) * 4
            ox2 = ox1 + (x2 - x1) * 4

            out_y1 = y1 * 4
            out_x1 = x1 * 4
            out_y2 = y2 * 4
            out_x2 = x2 * 4

            crop = result_tile[oy1:oy2, ox1:ox2]
            output[out_y1:out_y2, out_x1:out_x2] += crop.astype(np.float64)
            weights[out_y1:out_y2, out_x1:out_x2] += 1.0

            done += 1
            print(f"    Tile {done}/{total_tiles} done")

    weights = np.maximum(weights, 1e-8)
    output = (output / weights).astype(np.uint8)

    if scale == 2:
        output = cv2.resize(output, (w * 2, h * 2), interpolation=cv2.INTER_AREA)

    return output

def enhance_details(img, sharpen_level=0, grain_level=0):
    """Apply an unsharp mask and add subtle film grain to reduce Real-ESRGAN over-smoothing."""
    img_float = img.astype(np.float32)
    
    # Map sharpen_level (0-100) to amount (0.0 to 3.0) and radius
    if sharpen_level > 0:
        amount = (sharpen_level / 100.0) * 3.0
        radius = 1.0 + (sharpen_level / 100.0) * 2.0
        blurred = cv2.GaussianBlur(img_float, (0, 0), radius)
        img_float = float(amount + 1) * img_float - float(amount) * blurred
    
    # Map grain_level (0-100) to noise amplitude (0.0 to 15.0)
    if grain_level > 0:
        noise_amp = (grain_level / 100.0) * 15.0
        # Generate monochrome noise for natural film grain look
        noise = np.random.normal(0, noise_amp, img_float.shape[:2]).astype(np.float32)
        # Expand dims to add to all 3 channels
        noise = np.expand_dims(noise, axis=-1)
        img_float = img_float + noise

    sharpened = np.clip(img_float, 0, 255).astype(np.uint8)
    return sharpened


# ---- Flask Routes ----

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No files provided"}), 400

    scale = int(request.form.get("scale", 4))
    if scale not in (2, 4):
        scale = 4

    mode = request.form.get("mode", "upscale")
    if mode not in ("upscale", "colorize", "both"):
        mode = "upscale"

    batch_id = str(uuid.uuid4())[:8]
    results = []

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "error": f"Format not supported: {ext}"
            })
            continue

        file_id = f"{batch_id}_{len(results):03d}"
        input_path = UPLOAD_DIR / f"{file_id}{ext}"
        file.save(str(input_path))

        if input_path.stat().st_size > MAX_FILE_SIZE:
            input_path.unlink()
            results.append({
                "filename": file.filename,
                "error": "File too large (max 20MB)"
            })
            continue

        try:
            img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Cannot read image")

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            h, w = img.shape[:2]

            start = time.time()

            sharpen_level = int(request.form.get("sharpen_level", 0))
            grain_level = int(request.form.get("grain_level", 0))

            # Process based on mode
            if mode == 'colorize':
                output = colorize_image(img)
            elif mode == 'both':
                output = colorize_image(img)
                output = upscale_image(output, scale=scale)
                if sharpen_level > 0 or grain_level > 0:
                    output = enhance_details(output, sharpen_level=sharpen_level, grain_level=grain_level)
            else:  # upscale
                output = upscale_image(img, scale=scale)
                if sharpen_level > 0 or grain_level > 0:
                    output = enhance_details(output, sharpen_level=sharpen_level, grain_level=grain_level)

            elapsed = round(time.time() - start, 1)

            mode_suffix = f"_{mode}" if mode != 'upscale' else ''
            scale_suffix = f"_x{scale}" if mode != 'colorize' else ''
            out_name = f"{file_id}{mode_suffix}{scale_suffix}.png"
            output_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(output_path), output)

            oh, ow = output.shape[:2]

            results.append({
                "success": True,
                "filename": file.filename,
                "file_id": file_id,
                "download": f"/download/{out_name}",
                "original_size": f"{w}x{h}",
                "output_size": f"{ow}x{oh}",
                "scale": scale,
                "mode": mode,
                "time": elapsed,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify({
        "batch_id": batch_id,
        "results": results,
        "total": len(results),
        "success_count": sum(1 for r in results if r.get("success")),
    })


@app.route("/download/<filename>")
def download(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(path.resolve()), as_attachment=True)


@app.route("/download-zip", methods=["POST"])
def download_zip():
    """Download multiple results as a ZIP file."""
    data = request.get_json()
    filenames = data.get("files", [])
    if not filenames:
        return jsonify({"error": "No files specified"}), 400

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in filenames:
            # Sanitize filename
            safe_name = Path(fname).name
            path = OUTPUT_DIR / safe_name
            if path.exists():
                zf.write(str(path), safe_name)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='upscaled_images.zip'
    )


if __name__ == "__main__":
    print("[*] Image Upscaler running at http://localhost:5000")
    print("    First upload will download the AI model (~64MB)")
    print("    Using CPU - processing may take 30-60s per image")
    app.run(debug=False, port=5000)

