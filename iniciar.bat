@echo off
echo ================================
echo   Image Upscaler - Real-ESRGAN
echo ================================
echo.
echo Iniciando servidor...
echo Abre tu navegador en: http://127.0.0.1:5000
echo Presiona Ctrl+C para detener el servidor
echo.
cd /d "%~dp0"
venv\Scripts\python app.py
pause
