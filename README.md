Dataset utilizado se encuentra en : https://drive.google.com/drive/folders/1UuZ5f663PUS-a3UQ5gdicygFcSjh6Uru?usp=sharing
El sistema fue entrenado utilizando una fusión de datasets personalizados y optimizado para hardware NVIDIA Quadro.

## 📋 Características
- **Detección en tiempo real** usando webcam o archivos de video.
- **Modelo ligero:** Basado en `yolov8n` (Nano) para mayor velocidad.
- **Dataset Balanceado:** Combinación de datos de operarios con uniforme y personas con ropa de calle.
- **Inferencia optimizada** con OpenCV.

## ⚙️ Requisitos e Instalación

### Prerrequisitos
- Python 3.8 o superior
- Tarjeta gráfica NVIDIA (Recomendado para entrenamiento)
- Drivers CUDA instalados

### Instalación de dependencias
Se recomienda usar un entorno virtual. Instala las librerías necesarias:

```bash
# 1. Instalar PyTorch con soporte para CUDA (Ajustar según versión de CUDA, si n se tiene grafica con CUDA entonces usar CPU, sera mas lento.)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 2. Instalar Ultralytics (YOLO) y OpenCV
pip install ultralytics opencv-python
