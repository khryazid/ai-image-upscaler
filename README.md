# Resize — Image Upscaler & Colorizer Local con IA

Aplicación web local en Python/Flask para escalado de imágenes con **Real-ESRGAN** (2x / 4x) y colorización automática con **ECCV16**, ejecutada 100% en tu equipo sin subir imágenes a terceros.

## Estructura del Proyecto

- `app.py` — Servidor Flask (`127.0.0.1:5000`), carga de modelos Real-ESRGAN (PyTorch / ONNX Runtime) y procesamiento por *tiles*
- `colorizer.py` — Módulo de colorización ECCV16 (PyTorch / ONNX Runtime)
- `templates/` y `static/` — Interfaz web y estilos
- `iniciar.bat` — Lanzador rápido en Windows
- `MANUAL.md` — Manual de usuario detallado
- `models/` — Carpeta local para los pesos de IA *(excluida de Git)*

## Stack Tecnológico

- Python 3.10+, Flask 3.1.1
- PyTorch / TorchVision, OpenCV (`opencv-python-headless`), Pillow, NumPy (<2)
- Real-ESRGAN, GFPGAN, BasicSR, ONNX Runtime (opcional)

## Reinstalación y Ejecución Local

### 1. Crear entorno virtual e instalar dependencias
```powershell
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
```

### 2. Modelos de IA (`models/`)
Por tamaño (>100 MB), la carpeta `models/` no se versiona en GitHub.
- Si ya la tienes en local, consérvala en `Resize/models/` (`RealESRGAN_x4plus.pth`, `RealESRGAN_x4plus.onnx`, `RealESRGAN_x4plus.onnx.data`, `colorization_eccv16.pth`).
- Si partes desde cero, `app.py` y `colorizer.py` descargan automáticamente los pesos `.pth` oficiales la primera vez que procesas una imagen.

### 3. Ejecutar
- En Windows: doble clic en `iniciar.bat`, o desde terminal:
```powershell
venv\Scripts\python app.py
```
Luego abre **http://127.0.0.1:5000**.

## Licencia

Distribuido bajo la licencia [MIT](LICENSE).
