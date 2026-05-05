import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import time
import random
import string

class ImageDataset(Dataset):
    def __init__(self, n=200, size=128, variant=1):
        super().__init__()
        self.n = n
        self.size = size
        self.variant = variant
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.font = ImageFont.load_default()
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        
        if self.variant == 1:
            text = "ABC"
            x = random.randint(10, self.size-40)
            y = random.randint(10, self.size-40)
            
        elif self.variant == 2:
            text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
            x, y = 30, 30
            
        elif self.variant == 3:
            length = random.randint(1, 10)
            text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=length))
            x, y = 30, 30
            
        elif self.variant == 4:
            length = random.randint(1, 10)
            text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=length))
            x = random.randint(10, self.size-40)
            y = random.randint(10, self.size-40)
        
        draw.text((x, y), text, fill=0, font=self.font)
        tensor = self.transform(image)
        return tensor, tensor

class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.bottleneck = nn.Linear(256*16*16, latent)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent, 256*16*16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x

if __name__ == '__main__':
    for variant in [1, 2, 3, 4]:
        print(f"\nTraining variant {variant}")
        
        dataset = ImageDataset(2000, 256, variant=variant)
        plt.imshow(dataset[0][0].squeeze(), cmap='gray')
        plt.title(f"Пример изображения вариант {variant}")
        plt.show()
        
        encoder = Encoder()
        decoder = Decoder()
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        encoder.to(device)
        decoder.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
        
        encoder.train()
        decoder.train()
        
        epochs = 10
        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, _ in dataLoader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                latent = encoder(imgs)
                output = decoder(latent)
                loss = criterion(imgs, output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataLoader)
            print(f"{epoch=}, {avg_loss=:.2f}")
        
        torch.save(encoder.state_dict(), f"encoder_variant{variant}.pth")
        torch.save(decoder.state_dict(), f"decoder_variant{variant}.pth")