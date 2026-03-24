from train_model import CyrillicCNN, create_dataloaders
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    storage = Path("./tmp")
    model_file = storage / "cyrillic_cnn.pth"
    source = Path("Cyrillic")  
    
    network = CyrillicCNN()
    network.load_state_dict(torch.load(model_file, map_location="cpu"))
    network.eval()

    _, _, test_loader, test_set = create_dataloaders(source, batch_size=16)
    
    with torch.no_grad():
        for k in range(10):
            idx = np.random.randint(0, len(test_set))
            img, true_val = test_set[idx]
            result = network(img.unsqueeze(0))
            _, guess = torch.max(result.data, 1)
            
            print(f"true = {true_val}")
            print(f"pred = {guess[0]}")
            
            plt.imshow(img.squeeze(0), cmap='gray')
            plt.title(f"true={true_val}, pred={guess[0]}")
            plt.show()
    
    total_count = 0.0
    correct_count = 0.0
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            outputs = network(batch_imgs)
            _, guesses = torch.max(outputs.data, 1)
            
            total_count += batch_labels.size(0)
            correct_count += (guesses == batch_labels).sum().item()
    
    print(f'accuracy on test: {100 * correct_count / total_count:.2f}%')

    plot_file = Path("./tmp/history.png")
    if plot_file.exists():
        picture = plt.imread(plot_file)
        plt.figure(figsize=(10, 6))
        plt.title("plot")
        plt.imshow(picture)
        plt.axis('off')
        plt.show()