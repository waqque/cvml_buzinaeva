from ultralytics import YOLO
from pathlib import Path
import cv2


model_path = Path("yolo26/weights/best.pt")
category_names = {0: "cube", 1: "neither", 2: "sphere"}

print("загрузка модельки")
detector = YOLO(model_path)

webcam = cv2.VideoCapture(0)

while True:

    _, image = webcam.read()
    
    detection_results = detector.predict(
        source=image,
        device="cpu",     
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False      
    )[0]

    objects = detection_results.boxes
    if objects is not None:
        for obj in objects:

            x_left, y_top, x_right, y_bottom = map(int, obj.xyxy[0].tolist())

            score = float(obj.conf[0])

            class_id = int(obj.cls[0])
            class_name = category_names.get(class_id, f"class_{class_id}")
            cv2.rectangle(image, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)
            
            label_text = f"{class_name} {score:.2f}"
            cv2.putText(image, label_text, (x_left, y_top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("кубики шарики", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
