import torch
import cv2
import time
from train_model import build_model, transform
from pathlib import Path

save_path = Path(__file__).parent

model = build_model()
model.load_state_dict(torch.load(save_path / "model.pth"))
model.eval()

def predict(frame):
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()

    label = "person" if prob > 0.5 else "no person"
    return label, prob


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("p"):
        t = time.perf_counter()
        label, confidence = predict(frame)
        print(f"Elapsed time: {time.perf_counter() - t}")
        print(label, confidence)

cap.release()
cv2.destroyAllWindows()