import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch import nn, optim
from pathlib import Path
from PIL import Image, ImageOps


class CyrillicDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_paths = []
        self.labels = []
        
        label_idx = 0
        for class_dir in sorted(data_path.glob("*")):
            for img_path in class_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(label_idx)
            label_idx += 1
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGBA')
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, (0, 0), image)
        image = ImageOps.invert(background)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CyrillicCNN(nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate_model(data_loader, model, criterion=nn.CrossEntropyLoss()):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def create_dataloaders(data_path, batch_size=16):
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(3, (0.03, 0.03), (0.9, 1.0)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = CyrillicDataset(data_path, train_transform)
    
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    
    train_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.125,
        stratify=train_labels,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(CyrillicDataset(data_path, test_transform), val_idx)
    test_dataset = Subset(CyrillicDataset(data_path, test_transform), test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, test_dataset


if __name__ == "__main__":
    data_path = Path("Cyrillic")
    
    save_path = Path("./tmp")
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "cyrillic_cnn.pth"

    train_loader, val_loader, test_loader, test_dataset = create_dataloaders(data_path)
    
    model = CyrillicCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    num_epochs = 15
  
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc = evaluate_model(val_loader, model)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    torch.save({
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
    }, save_path / "training_history.pt")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / "history.png")
    plt.show()