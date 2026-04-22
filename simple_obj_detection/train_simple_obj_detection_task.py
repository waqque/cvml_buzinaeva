import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

classes = ["square", "circle", "triangle"]


class ShapeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data = []
        
        for idx, cls_name in enumerate(classes):
            img_dir = data_path / cls_name / "images"
            label_dir = data_path / cls_name / "labels"
            
            for img_file in sorted(img_dir.glob("*.png")):
                label_file = label_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    self.data.append((img_file, label_file, idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label_path, cls_id = self.data[index]
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img_tensor = self.transform(Image.fromarray(img))
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        values = label_path.read_text().strip().split()
        _, cx, cy, w, h = map(float, values)
        bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        
        return img_tensor, cls_id, bbox


class Detector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_layers(x)
        cls_out = self.classifier(features)
        bbox_out = self.bbox_regressor(features)
        return cls_out, bbox_out


def compute_iou(box1, box2):
    x1_1 = box1[0] - box1[2]/2
    y1_1 = box1[1] - box1[3]/2
    x2_1 = box1[0] + box1[2]/2
    y2_1 = box1[1] + box1[3]/2
    
    x1_2 = box2[0] - box2[2]/2
    y1_2 = box2[1] - box2[3]/2
    x2_2 = box2[0] + box2[2]/2
    y2_2 = box2[1] + box2[3]/2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter_area
    
    return inter_area / union if union > 0 else 0.0


def giou_loss(pred_box, target_box):
    p_x1 = pred_box[:, 0] - pred_box[:, 2] / 2
    p_y1 = pred_box[:, 1] - pred_box[:, 3] / 2
    p_x2 = pred_box[:, 0] + pred_box[:, 2] / 2
    p_y2 = pred_box[:, 1] + pred_box[:, 3] / 2
    
    t_x1 = target_box[:, 0] - target_box[:, 2] / 2
    t_y1 = target_box[:, 1] - target_box[:, 3] / 2
    t_x2 = target_box[:, 0] + target_box[:, 2] / 2
    t_y2 = target_box[:, 1] + target_box[:, 3] / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    p_area = ((p_x2 - p_x1).clamp(min=0)) * ((p_y2 - p_y1).clamp(min=0))
    t_area = ((t_x2 - t_x1).clamp(min=0)) * ((t_y2 - t_y1).clamp(min=0))
    union_area = p_area + t_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    c_area = ((c_x2 - c_x1).clamp(min=0)) * ((c_y2 - c_y1).clamp(min=0))
    
    giou = iou - (c_area - union_area) / (c_area + 1e-6)
    return (1 - giou).mean()


def total_loss(cls_pred, bbox_pred, cls_true, bbox_true, lambda_box=8.0):
    loss_cls = F.cross_entropy(cls_pred, cls_true)
    loss_giou = giou_loss(bbox_pred, bbox_true)
    loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_true)
    
    total_bbox_loss = loss_giou + 0.5 * loss_bbox
    total = loss_cls + lambda_box * total_bbox_loss
    
    return total, loss_cls, total_bbox_loss


def visualize_results(data_loader, model, save_folder, num_samples=8):
    model.eval()
    images, true_classes, true_boxes = next(iter(data_loader))
    
    with torch.no_grad():
        pred_classes, pred_boxes = model(images)
    
    pred_labels = pred_classes.argmax(1)
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(14, 7))
    
    iou_scores = []
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
            
        img_np = images[i].numpy().transpose(1, 2, 0)
        h, w = img_np.shape[:2]
        
        cx, cy, bw, bh = true_boxes[i].numpy()
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        ax.add_patch(Rectangle((x1, y1), bw*w, bh*h, fill=False, edgecolor='blue', linewidth=2))
        
        cx, cy, bw, bh = pred_boxes[i].numpy()
        x1_pred = (cx - bw/2) * w
        y1_pred = (cy - bh/2) * h
        ax.add_patch(Rectangle((x1_pred, y1_pred), bw*w, bh*h, fill=False, edgecolor='red', linewidth=2))
        
        iou = compute_iou(true_boxes[i].numpy(), pred_boxes[i].numpy())
        iou_scores.append(iou)
        
        is_correct = pred_labels[i] == true_classes[i]
        color = 'green' if is_correct else 'red'
        ax.set_title(f"True: {classes[true_classes[i]]} | Pred: {classes[pred_labels[i]]} | IoU: {iou:.2f}", 
                     color=color, fontsize=9)
        ax.imshow(img_np)
        ax.axis('off')
    
    plt.suptitle(f"Avg IoU: {np.mean(iou_scores):.3f}")
    plt.tight_layout()
    plt.savefig(save_folder / "prediction_examples.png", dpi=150)
    plt.show()


def run_experiment(data_root, epochs_count=40):
    best_model_path = data_root / "best_model.pt"
    if best_model_path.exists():
        best_model_path.unlink()
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_set = ShapeDataset(data_root / "train", transform=transform)
    val_set = ShapeDataset(data_root / "val", transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    
    model = Detector(num_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_accuracy = 0.0
    best_iou = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_iou': []}
    
    print(f"\n=== {data_root.name} ===")
    
    for epoch in range(1, epochs_count + 1):
        model.train()
        epoch_loss = 0.0
        
        for images, cls_true, box_true in train_loader:
            optimizer.zero_grad()
            cls_pred, box_pred = model(images)
            loss, loss_cls, loss_box = total_loss(cls_pred, box_pred, cls_true, box_true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        iou_list = []
        
        with torch.no_grad():
            for images, cls_true, box_true in val_loader:
                cls_pred, box_pred = model(images)
                loss, loss_cls, loss_box = total_loss(cls_pred, box_pred, cls_true, box_true)
                val_loss += loss.item()
                
                predicted = cls_pred.argmax(1)
                correct += (predicted == cls_true).sum().item()
                total += cls_true.size(0)
                
                for j in range(len(box_true)):
                    iou = compute_iou(box_true[j].numpy(), box_pred[j].numpy())
                    iou_list.append(iou)
        
        val_accuracy = correct / total
        avg_iou = np.mean(iou_list)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_iou'].append(avg_iou)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_iou = avg_iou
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss: {history['train_loss'][-1]:.4f} | val_loss: {avg_val_loss:.4f} | acc: {val_accuracy:.3f} | iou: {avg_iou:.3f}")
        
        if val_accuracy >= 0.95 and avg_iou >= 0.85:
            print(f"Target achieved at epoch {epoch}")
            break
        
        if patience_counter >= 12:
            print(f"Early stop at epoch {epoch}")
            break
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['val_acc'], label='accuracy', color='green')
    axes[1].axhline(y=0.95, color='r', linestyle='--')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(history['val_iou'], label='iou', color='orange')
    axes[2].axhline(y=0.85, color='r', linestyle='--')
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('iou')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.suptitle(data_root.name)
    plt.tight_layout()
    plt.savefig(data_root / "curves.png", dpi=150)
    plt.show()
    
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        visualize_results(val_loader, model, data_root)
    
    print(f"Best: acc={best_accuracy:.3f}, iou={best_iou:.3f}\n")
    
    return best_accuracy, best_iou


base_path = Path(".")

dataset_paths = {
    "clean": base_path / "shapes_dataset",
    "with_bg": base_path / "shapes_dataset_bg", 
    "random": base_path / "shapes_dataset_random"
}

results = {}
for name, path in dataset_paths.items():
    if path.exists():
        acc, iou = run_experiment(path, epochs_count=40)
        results[name] = {'accuracy': acc, 'iou': iou}

for name, metrics in results.items():
    print(f"{name:12s}: acc={metrics['accuracy']:.3f}, iou={metrics['iou']:.3f}")