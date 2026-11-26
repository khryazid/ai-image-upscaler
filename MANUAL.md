# Image Upscaler - Manual de Usuario

## Que es esto?

Una app local que usa inteligencia artificial (Real-ESRGAN) para aumentar el tamano y mejorar la calidad de tus imagenes. Todo corre en tu PC, no sube nada a internet.

---

## Como iniciar

### Opcion 1: Doble clic (recomendado)

1. Ve a la carpeta del proyecto (`ai-image-upscaler` / `Resize`)
2. Haz doble clic en **`iniciar.bat`**
3. Se abrira una ventana negra con el mensaje del servidor
4. Abre tu navegador en: **http://127.0.0.1:5000**

### Opcion 2: Desde terminal

```
cd ai-image-upscaler
venv\Scripts\python app.py
```

Luego abre **http://127.0.0.1:5000** en tu navegador.

---

## Como usar la app

### 1. Subir imagen
- **Arrastra** una imagen sobre la zona de drop, o
- **Haz clic** en la zona para buscar un archivo
- Formatos soportados: PNG, JPG, WebP, BMP, TIFF
- Tamano maximo: 20MB

### 2. Elegir escala
- **2x** - Duplica el tamano (mas rapido)
- **4x** - Cuadruplica el tamano (mas lento, mejor resultado)

### 3. Procesar
- Haz clic en el boton **Upscale**
- La primera vez descargara el modelo de IA (~64MB). Solo pasa una vez.
- Espera ~30-60 segundos (depende del tamano de la imagen)

### 4. Descargar
- Veras una comparacion antes/despues
- Haz clic en **Download** para guardar el resultado en PNG

---

## Detener el servidor

- Cierra la ventana negra del terminal, o
- Presiona **Ctrl+C** en la terminal

---

## Tiempos estimados de procesamiento (CPU)

| Tamano original | Escala | Tiempo aprox |
|-----------------|--------|--------------|
| 256x256         | 4x     | ~5 seg       |
| 512x512         | 4x     | ~15 seg      |
| 1024x1024       | 4x     | ~60 seg      |
| 1920x1080       | 4x     | ~2-3 min     |

> Nota: Imagenes muy grandes se procesan en tiles (fragmentos) para no quedarse sin memoria.

---

## Estructura de archivos

```
Resize/
  app.py            <- Servidor backend
  iniciar.bat       <- Lanzador (doble clic)
  venv/             <- Entorno virtual Python
  models/           <- Modelo IA (se descarga automaticamente)
  uploads/          <- Imagenes originales subidas
  outputs/          <- Imagenes procesadas (resultado)
  templates/        <- Interfaz web (HTML)
  static/           <- Estilos CSS
```

---

## Solucion de problemas

| Problema | Solucion |
|----------|----------|
| "localhost refused to connect" | Usa `http://127.0.0.1:5000` en vez de `localhost:5000` |
| La descarga del modelo falla | Verifica tu conexion a internet. El modelo se descarga de GitHub |
| Procesamiento muy lento | Normal en CPU. Imagenes grandes toman mas tiempo |
| Error de memoria | Usa imagenes mas pequenas o escala 2x |
| Puerto 5000 ocupado | Cierra otras apps que usen ese puerto, o cambia el puerto en `app.py` (linea final) |
