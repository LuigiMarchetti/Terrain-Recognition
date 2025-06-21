import cv2
import numpy as np
import matplotlib.pyplot as plt


def calcular_excesso_vegetacao(img):
    img = img.astype(np.float32)
    B, G, R = cv2.split(img)
    exg = 2 * G - R - B
    return exg


def detectar_desmatamento(img_antes, img_depois, threshold=30):
    exg_antes = calcular_excesso_vegetacao(img_antes)
    exg_depois = calcular_excesso_vegetacao(img_depois)
    diferenca = exg_depois - exg_antes
    desmatamento = (diferenca < -threshold).astype(np.uint8) * 255
    return desmatamento


def remover_manchas_pequenas(mascara, min_area=150):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nova_mascara = np.zeros_like(mascara)

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(nova_mascara, [cnt], -1, 255, -1)

    return nova_mascara


def aplicar_mascara_em_vermelho(img, mascara):
    img_com_mascara = img.copy()
    vermelho = [0, 0, 255]
    mask_indices = mascara > 0
    img_com_mascara[mask_indices] = vermelho
    return img_com_mascara


def calcular_percentual_desmatado(mascara):
    total_pixels = mascara.size
    pixels_desmatados = np.count_nonzero(mascara)
    percentual = (pixels_desmatados / total_pixels) * 100
    return percentual


# Carrega as imagens
img_2022 = cv2.imread('sentinel/resultados_ndvi/rgb_passado_2022-06-13.png')
img_2024 = cv2.imread('sentinel/resultados_ndvi/rgb_atual_2025-03-09.png')

# Redimensiona se necessário
img_2022 = cv2.resize(img_2022, (img_2024.shape[1], img_2024.shape[0]))

# Detecta e limpa a máscara
mascara_bruta = detectar_desmatamento(img_2022, img_2024)
mascara_limpa = remover_manchas_pequenas(mascara_bruta, min_area=50)

# Calcula percentual
percentual = calcular_percentual_desmatado(mascara_limpa)
print(f"Área desmatada (após limpeza): {percentual:.2f}%")

# Aplica a máscara vermelha sobre a imagem de 2024
img_destacada = aplicar_mascara_em_vermelho(img_2024, mascara_limpa)

# Salva as imagens
cv2.imwrite('mascara_desmatamento_limpa.png', mascara_limpa)
cv2.imwrite('desmatamento_destacado_limpo.png', img_destacada)

# Exibe resultado
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Imagem recente com Máscara Vermelha")
plt.imshow(cv2.cvtColor(img_destacada, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Máscara Bruta")
plt.imshow(mascara_bruta, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Máscara Filtrada (Limpa)")
plt.imshow(mascara_limpa, cmap='gray')
plt.axis('off')

plt.suptitle(f"Área desmatada filtrada: {percentual:.2f}%", fontsize=16)
plt.tight_layout()
plt.show()
