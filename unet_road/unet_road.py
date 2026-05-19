import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import time
import sys

path = Path("roads")

class RoadsDataset(Dataset):
    def __init__(self, path, target_size=(256, 256)):  # уменьшила размер для ускорения
        super().__init__()
        self.images_path = path / "images"
        self.masks_path = path / "masks"
        self.target_size = target_size
        self.images = list(self.images_path.glob("*.png"))
        self.masks = list(self.masks_path.glob("*.png"))
    
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.target_size, Image.Resampling.NEAREST)
        
        image = np.array(image, dtype=np.float32) / 255.
        mask = np.array(mask, dtype=np.float32)
        mask = (mask == 82).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)
        
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()
        
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask = torch.from_numpy(mask)
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for n in features:
            self.downscale.append(DoubleConv(in_channels, n))
            in_channels = n

        for n in reversed(features):
            self.upscale.append(nn.ConvTranspose2d(n * 2, n, 2, 2))
            self.upscale.append(DoubleConv(n * 2, n))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.result = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for ds in self.downscale:
            x = ds(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]
        
        for idx in range(0, len(self.upscale), 2):
            x = self.upscale[idx](x)
            skip = skips[idx // 2]
            if x.shape[2:] != skip.shape[2:]:
                skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            cx = torch.cat((skip, x), dim=1)
            x = self.upscale[idx + 1](cx)
        return self.result(x)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        p_area = pred_sig.view(-1)
        t_area = target.view(-1)
        intersection = (p_area * t_area).sum()
        return 1 - (2 * intersection + 1) / (p_area.sum() + t_area.sum() + 1)

def train_model(model, train_loader, epochs=30, lr=0.001, device='cpu'):
    print(f"Training on device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DiceLoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 2 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}", flush=True)
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}', flush=True)
    
    return model

if __name__ == "__main__":
    dataset = RoadsDataset(path, target_size=(256, 256))  
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)  

    model = UNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_images, test_masks = next(iter(train_loader))

    start_time = time.time()

    trained_model = train_model(model, train_loader, epochs=30, lr=0.001, device=device)    
    training_time = time.time() - start_time

    torch.save(trained_model.state_dict(), 'unet_road_model.pth')
