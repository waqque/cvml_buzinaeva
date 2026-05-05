from train import Encoder, Decoder, ImageDataset
import torch
import matplotlib.pyplot as plt

for variant in [1, 2, 3, 4]:
    encoder = Encoder()
    decoder = Decoder()
    
    encoder.load_state_dict(torch.load(f"encoder_variant{variant}.pth"))
    decoder.load_state_dict(torch.load(f"decoder_variant{variant}.pth"))
    
    encoder.eval()
    decoder.eval()
    
    dataset = ImageDataset(10, 256, variant=variant)
    image, _ = dataset[0]
    
    with torch.no_grad():
        latent = encoder(image.unsqueeze(0))
        result = decoder(latent)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Original (var {variant})')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(result.squeeze(), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow((image.squeeze() - result.squeeze()), cmap='gray')
    plt.title('Difference')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()