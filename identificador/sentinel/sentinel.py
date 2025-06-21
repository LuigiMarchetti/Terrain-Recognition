from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, SentinelHubCatalog
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from PIL import Image
import os

# === CONFIGURAÇÃO DO SENTINEL HUB ===
config = SHConfig()
config.sh_client_id = ''
config.sh_client_secret = ''

# Coordenada e área
lat, lon = -26.90567708545667, -49.0556474962708
aoi = BBox([lon - 0.0135, lat - 0.0135, lon + 0.0135, lat + 0.0135], crs=CRS.WGS84)
resolution = 10
width, height = bbox_to_dimensions(aoi, resolution)

# Datas
hoje = date.today()
tres_anos_atras = hoje - timedelta(days=3 * 365)

# === BUSCAR IMAGEM COM POUCA NUVEM ===
catalog = SentinelHubCatalog(config=config)

def buscar_data_valida(data_inicio, data_fim):
    search_iterator = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=aoi,
        time=(data_inicio, data_fim),
        filter='eo:cloud_cover < 20',
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
        limit=10
    )
    results = list(search_iterator)
    if not results:
        return None
    results.sort(key=lambda x: x["properties"]["eo:cloud_cover"])
    dt_str = results[0]["properties"]["datetime"]
    dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    return dt_obj.date()

data_atual = buscar_data_valida(hoje - timedelta(days=120), hoje)
data_passada = buscar_data_valida(tres_anos_atras - timedelta(days=360), tres_anos_atras)

if not data_atual or not data_passada:
    print("Não foi possível encontrar imagens com baixa cobertura de nuvem.")
    exit()

print(f"Imagem atual: {data_atual}")
print(f"Imagem passada: {data_passada}")

# === EVALSCRIPT RGB BRUTA (sem NDVI, salva como imagem visual) ===
evalscript_rgb = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"], // Vermelho, Verde, Azul
    output: { bands: 3 }
  };
}

// Fatores de ganho e gama para realçar a imagem
const gain = 2.5;
const gamma = 1.3;

function evaluatePixel(sample) {
  return [
    gain * Math.pow(sample.B04, 1/gamma),
    gain * Math.pow(sample.B03, 1/gamma),
    gain * Math.pow(sample.B02, 1/gamma)
  ];
}
"""

def requisicao_rgb(data):
    return SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(data, data)
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi,
        size=(width, height),
        config=config
    )

# === EVALSCRIPT NDVI COM MÁSCARA DE NUVEM ===
evalscript_ndvi = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "SCL"],
    output: { bands: 1 }
  };
}
function evaluatePixel(sample) {
  if ([3, 8, 9, 10].includes(sample.SCL)) {
    return [0]; // mascarar nuvem
  }
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi];
}
"""

def requisicao_ndvi(data):
    return SentinelHubRequest(
        evalscript=evalscript_ndvi,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(data, data)
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi,
        size=(width, height),
        config=config
    )

output_folder = "resultados_ndvi"
os.makedirs(output_folder, exist_ok=True)
# Executar as requisições
print("Baixando imagens...")
ndvi_atual = requisicao_ndvi(data_atual).get_data()[0].squeeze()
ndvi_passado = requisicao_ndvi(data_passada).get_data()[0].squeeze()
rgb_passado = requisicao_rgb(data_passada).get_data()[0]  # (H, W, 3)
rgb_atual = requisicao_rgb(data_atual).get_data()[0]

# Garantir conversão correta
rgb_passado_uint8 = np.clip(rgb_passado / np.max(rgb_passado) * 255, 0, 255).astype(np.uint8)
rgb_atual_uint8 = np.clip(rgb_atual / np.max(rgb_atual) * 255, 0, 255).astype(np.uint8)

# Salvar diretamente, sem mover eixo
Image.fromarray(rgb_passado_uint8).save(f"{output_folder}/rgb_passado_{data_passada}.png")
Image.fromarray(rgb_atual_uint8).save(f"{output_folder}/rgb_atual_{data_atual}.png")

# === NDVI COMPARAÇÃO ===
delta_ndvi = ndvi_atual - ndvi_passado
mascara_perda = (delta_ndvi < -0.008).astype(np.uint8)

# === CALCULAR PORCENTAGEM ===
percentual_perda = (np.count_nonzero(mascara_perda) / mascara_perda.size) * 100
print(f"Perda de vegetação estimada: {percentual_perda:.2f}%")

# === SALVAR RESULTADOS ===

# Salvar NDVIs
Image.fromarray(((ndvi_passado + 1) / 2 * 255).astype(np.uint8)).save(f"{output_folder}/ndvi_passado.png")
Image.fromarray(((ndvi_atual + 1) / 2 * 255).astype(np.uint8)).save(f"{output_folder}/ndvi_atual.png")

# Salvar máscara
Image.fromarray((mascara_perda * 255).astype(np.uint8)).save(f"{output_folder}/mascara_perda.png")

# === VISUALIZAÇÃO ===
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title(f"NDVI {data_passada}")
plt.imshow(ndvi_passado, cmap='YlGn')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title(f"NDVI {data_atual}")
plt.imshow(ndvi_atual, cmap='YlGn')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title(f"Perda Vegetação ({percentual_perda:.2f}%)")
plt.imshow(mascara_perda, cmap='Reds')
plt.tight_layout()
plt.show()
