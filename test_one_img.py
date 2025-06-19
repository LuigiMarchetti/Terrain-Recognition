import tifffile
import matplotlib.pyplot as plt

# Abrir a imagem .tif
img = tifffile.imread("./EuroSAT_MS/Forest/Forest_1.tif")  # shape: (bandas, altura, largura) ou (altura, largura, bandas)

# Ver forma da imagem
print(f"Formato da imagem: {img.shape}")  # útil para saber se está (bands, H, W) ou (H, W, bands)

# Exibir todas as bandas individualmente
num_bandas = img.shape[0] if img.shape[0] < 20 else img.shape[-1]  # Heurística

for i in range(num_bandas):
    plt.imshow(img[i], cmap='gray') if img.shape[0] < 20 else plt.imshow(img[:, :, i], cmap='gray')
    plt.title(f"Banda {i + 1}")
    plt.axis('off')
    plt.show()
