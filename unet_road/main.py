import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from unet_road import RoadsDataset, UNet, path

print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('unet_road_model.pth', map_location=device))
model.eval()
print("Model loaded successfully")


ds = RoadsDataset(path)

idx = np.random.randint(0, len(ds))
image, true_mask = ds[0]
print(f"Showing example {idx+1}/{len(ds)}")


with torch.no_grad():
    image_batch = image.unsqueeze(0).to(device)
    pred = model(image_batch)
    pred_mask = torch.sigmoid(pred)
    pred_mask = (pred_mask > 0.5).float()
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()

image_np = image.numpy().transpose(1, 2, 0)
true_mask_np = true_mask.squeeze(0).numpy()

diff_mask = np.abs(true_mask_np - pred_mask)

fig, ax = plt.subplots(1, 4, figsize=(16, 4))

ax[0].imshow(image_np)
ax[0].set_title('Original Image')
# ax[0].axis('off')

ax[1].imshow(true_mask_np, cmap='gray')
ax[1].set_title('Mask')
# ax[1].axis('off')

ax[2].imshow(pred_mask, cmap='gray')
ax[2].set_title('Predicted Mask')
# ax[2].axis('off')

ax[3].imshow(diff_mask, cmap='hot')
ax[3].set_title('Difference')
# ax[3].axis('off')

plt.tight_layout()
plt.savefig('result.png', dpi=150)
plt.show()

intersection = (true_mask_np * pred_mask).sum()
union = (true_mask_np + pred_mask).sum()
iou = intersection / union if union > 0 else 0

accuracy = (true_mask_np == pred_mask).sum() / true_mask_np.size

print(f"  IoU: {iou:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Result saved as result.png")