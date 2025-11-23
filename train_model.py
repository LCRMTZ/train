from ultralytics import YOLO

# --- ESTA LÍNEA ES OBLIGATORIA EN WINDOWS ---
if __name__ == '__main__':
    
    # Cargas el modelo base
    model = YOLO("yolov8n.pt")

    # Entrenas
    model.train(
        data="dataset/data.yaml", 
        epochs=50,
        imgsz=640,
        batch=8,       
        device=0,      
        workers=4      
    )