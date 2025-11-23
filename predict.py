import cv2
from ultralytics import YOLO

# Cargar tu modelo entrenado
model = YOLO("best.pt")

# Ruta de la imagen a analizar
image_path = "worker.webp"   # ← cámbiala por la imagen que quieras detectar

# Leer la imagen
img = cv2.imread(image_path)

# Redimensionar (opcional pero recomendado)
img_resized = cv2.resize(img, (640, 640))

# Inferencia
results = model(img_resized)

# Procesar detecciones
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        print(f"Detectado: {label} ({conf:.2f})")

    # Dibujar cajas en la imagen
    annotated = r.plot()

# Mostrar resultado
cv2.imshow("Resultados", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
