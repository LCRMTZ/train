import cv2
from ultralytics import YOLO

# --- PASO 1: CARGAR EL MODELO ---
# Si "best.pt" no está en la misma carpeta, pon la ruta completa.
# Ejemplo: "runs/detect/train/weights/best.pt"
try:
    model = YOLO("best.pt") 
except:
    print("¡ERROR! No encuentro 'best.pt'. Asegúrate de copiarlo a esta carpeta.")
    exit()

# --- PASO 2: ABRIR CÁMARA ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- PASO 3: INFERENCIA ---
    # No hace falta hacer cv2.resize manual, YOLO lo maneja internamente
    # conf=0.6 significa: "Solo muéstrame si estás 60% seguro"
    results = model(frame, stream=True, conf=0.6)

    # Procesar resultados
    for r in results:
        
        # (Opcional) Imprimir en consola qué detectó
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confianza = float(box.conf[0])
            print(f"DETECTADO: {label} ({confianza:.2f})")

        # Dibujar las cajas en la imagen
        annotated_frame = r.plot()

        # Mostrar la imagen en una ventana
        cv2.imshow("Detector de Uniformes", annotated_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()