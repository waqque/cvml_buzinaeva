import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from pathlib import Path
from collections import deque
import cv2
import time

save_path = Path(__file__).parent

def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    return model

model = build_model()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

model_path = save_path / "model.pth"
if model_path.exists():
    model.load_state_dict(torch.load(model_path))
    print("модель, обученная ранее, загружена")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class Buffer():
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)

    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        return images, labels

def train(buffer):
    if len(buffer) < 10:
        return None

    model.train()
    images, labels = buffer.get_batch()

    optimizer.zero_grad()
    predictions = model(images).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def save_model():
    torch.save(model.state_dict(), model_path)
    print("Модель сохранена")

def run_training():
    cap = cv2.VideoCapture(0)
    buffer = Buffer()
    count_labeled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if key == ord("q"):#quit
            break

        elif key == ord("1"):  # person
            tensor = transform(image)
            buffer.append(tensor, 1.0)
            count_labeled += 1

        elif key == ord("2"):  # no person
            tensor = transform(image)
            buffer.append(tensor, 0.0)
            count_labeled += 1

        elif key == ord("s"): #save
            save_model()

        if count_labeled >= buffer.frames.maxlen:
            loss = train(buffer)
            if loss:
                print(f"Loss = {loss}")
            count_labeled = 0

    cap.release()
    cv2.destroyAllWindows()
    save_model()


if __name__ == "__main__":
    run_training()