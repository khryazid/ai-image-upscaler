"""
Colorization module using the ECCV16 model by Richard Zhang et al.
Pure PyTorch implementation - no extra dependencies.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
    has_ort = True
except ImportError:
    has_ort = False

try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    device = torch.device('cpu')

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

COLORIZE_MODEL_URL = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"
COLORIZE_MODEL_PATH = MODEL_DIR / "colorization_eccv16.pth"
COLORIZE_ONNX_PATH = MODEL_DIR / "colorization_eccv16.onnx"

_colorizer = None
_ort_session = None


# ---- ECCV16 Colorization Network ----

class ECCVGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

        norm = nn.BatchNorm2d

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True), nn.ReLU(True),
            norm(64),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True), nn.ReLU(True),
            norm(128),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
            norm(256),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True),
            norm(512),
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
            norm(512),
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
            norm(512),
        )
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True),
            norm(512),
        )
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 313, 1, 1, 0, bias=True),
        )

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, 1, 1, 0, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, input_l):
        # Normalize L channel
        x = (input_l - self.l_cent) / self.l_norm

        conv1 = self.model1(x)
        conv2 = self.model2(conv1[:, :, ::2, ::2])  # Downsample 2x
        conv3 = self.model3(conv2[:, :, ::2, ::2])  # Downsample 4x
        conv4 = self.model4(conv3[:, :, ::2, ::2])  # Downsample 8x
        conv5 = self.model5(conv4)
        conv6 = self.model6(conv5)
        conv7 = self.model7(conv6)
        conv8 = self.model8(conv7)

        out = self.model_out(self.softmax(conv8))

        # Unnormalize AB and upsample
        return self.upsample4(out) * self.ab_norm


# ---- Load & Colorize ----

def download_colorize_model():
    """Download colorization model if not present."""
    if COLORIZE_MODEL_PATH.exists():
        return
    print("[*] Downloading colorization model (~123MB)...")
    print(f"    From: {COLORIZE_MODEL_URL}")
    urllib.request.urlretrieve(COLORIZE_MODEL_URL, str(COLORIZE_MODEL_PATH))
    size_mb = COLORIZE_MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"[OK] Colorization model downloaded ({size_mb:.1f} MB)")


def get_colorizer():
    """Lazy-load the colorization model."""
    global _colorizer, _ort_session
    if _ort_session is not None: return _ort_session
    if _colorizer is not None: return _colorizer
    
    download_colorize_model()
    
    if has_ort:
        if not COLORIZE_ONNX_PATH.exists():
            print("[*] Converting Colorizer to ONNX format...")
            model = ECCVGenerator()
            state_dict = torch.load(str(COLORIZE_MODEL_PATH), map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            
            dummy_input = torch.randn(1, 1, 256, 256)
            torch.onnx.export(model, dummy_input, str(COLORIZE_ONNX_PATH),
                              opset_version=14,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                            'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
            print("[OK] Colorizer ONNX conversion complete")
            del model
            
        print("[*] Loading Colorizer ONNX session with DirectML")
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        _ort_session = ort.InferenceSession(str(COLORIZE_ONNX_PATH), providers=providers)
        return _ort_session

    model = ECCVGenerator()
    state_dict = torch.load(str(COLORIZE_MODEL_PATH), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _colorizer = model
    print("[OK] Colorization model loaded and ready")
    return _colorizer


def colorize_image(img_bgr):
    """
    Colorize a grayscale or B&W image.
    Input: HWC BGR uint8
    Output: HWC BGR uint8
    """
    model_or_session = get_colorizer()
    h, w = img_bgr.shape[:2]

    # Convert to LAB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L_orig = img_lab[:, :, 0]  # Original L channel (full resolution)

    # Resize L to 256x256 for model input
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_lab_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    L_resized = img_lab_resized[:, :, 0]

    if has_ort and hasattr(model_or_session, 'run'):
        L_input = L_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)
        ort_inputs = {model_or_session.get_inputs()[0].name: L_input}
        ab_output = model_or_session.run(None, ort_inputs)[0]
        ab = ab_output[0].transpose(1, 2, 0) # HxWx2
    else:
        # To tensor: 1xHxW -> 1x1xHxW
        L_tensor = torch.from_numpy(L_resized).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            ab_output = model_or_session(L_tensor)
        ab = ab_output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HxWx2

    # Get AB prediction and resize to original size
    ab_resized = cv2.resize(ab, (w, h))

    # Combine original L with predicted AB
    lab_result = np.zeros((h, w, 3), dtype=np.float32)
    lab_result[:, :, 0] = L_orig
    lab_result[:, :, 1] = ab_resized[:, :, 0]
    lab_result[:, :, 2] = ab_resized[:, :, 1]

    # Convert back to BGR
    rgb_result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2RGB)
    rgb_result = np.clip(rgb_result * 255, 0, 255).astype(np.uint8)
    bgr_result = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR)

    return bgr_result
