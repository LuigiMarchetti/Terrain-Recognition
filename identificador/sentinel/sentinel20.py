from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, SentinelHubCatalog
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from PIL import Image
import os
import pandas as pd

# === CONFIGURAÇÃO DO SENTINEL HUB ===
config = SHConfig()
config.sh_client_id = ''
config.sh_client_secret = ''

# === LISTA DE COORDENADAS ===
coordenadas = [
    #(-26.90567708545667, -49.0556474962708),
    (-26.96046686848641, -49.145220530125506)
    #(-26.868825565055406, -49.16984456369647)
    # adicione mais coordenadas aqui se quiser
]

limiar_mascara_vegetacao = 0.4
limiar_mascara_perda = -0.5

hoje = date.today()
tres_anos_atras = hoje - timedelta(days=3 * 365)

# === EVALSCRIPTS ===
evalscript_rgb = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}
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

evalscript_ndvi = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "SCL"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  if ([3, 6, 8, 9, 10].includes(sample.SCL)) {
    return [null];
  }
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi];
}
"""

def buscar_data_valida(catalog, aoi, data_inicio, data_fim):
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

def requisicao_rgb(data, aoi, width, height):
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

def requisicao_ndvi(data, aoi, width, height):
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

def classificar_perda(p):
    if p < 1:
        return "Inalterado"
    elif p < 5:
        return "Alerta"
    elif p < 10:
        return "Grave"
    else:
        return "Gravíssimo"

# === ARMAZENAR RESULTADOS ===
resultados = []

# === PROCESSAR TODAS AS COORDENADAS ===
for idx, (lat, lon) in enumerate(coordenadas):
    print(f"\nProcessando coordenada {idx+1}: ({lat}, {lon})")

    aoi = BBox([lon - 0.0050, lat - 0.0050, lon + 0.0050, lat + 0.0050], crs=CRS.WGS84)
    width, height = bbox_to_dimensions(aoi, 10)

    catalog = SentinelHubCatalog(config=config)
    data_atual = buscar_data_valida(catalog, aoi, hoje - timedelta(days=120), hoje)
    data_passada = buscar_data_valida(catalog, aoi, tres_anos_atras - timedelta(days=360), tres_anos_atras)

    if not data_atual or not data_passada:
        print("  Imagens com baixa cobertura de nuvem não encontradas.")
        continue

    print(f"  Imagem atual: {data_atual} | Imagem passada: {data_passada}")

    ndvi_atual = requisicao_ndvi(data_atual, aoi, width, height).get_data()[0].squeeze()
    ndvi_passado = requisicao_ndvi(data_passada, aoi, width, height).get_data()[0].squeeze()

    rgb_atual = requisicao_rgb(data_atual, aoi, width, height).get_data()[0]
    rgb_passado = requisicao_rgb(data_passada, aoi, width, height).get_data()[0]

    rgb_atual_uint8 = np.clip(rgb_atual / np.max(rgb_atual) * 255, 0, 255).astype(np.uint8)
    rgb_passado_uint8 = np.clip(rgb_passado / np.max(rgb_passado) * 255, 0, 255).astype(np.uint8)

    delta_ndvi = ndvi_atual - ndvi_passado
    mascara_veg = ndvi_passado > limiar_mascara_vegetacao
    mascara_perda = (delta_ndvi < limiar_mascara_perda) & mascara_veg

    pixels_validos = np.count_nonzero(mascara_veg)
    pixels_perda = np.count_nonzero(mascara_perda)
    percentual = (pixels_perda / pixels_validos) * 100 if pixels_validos else 0
    classificacao = classificar_perda(percentual)

    pasta = f"resultados_ndvi/{lat:.5f}_{lon:.5f}"
    os.makedirs(pasta, exist_ok=True)

    Image.fromarray(rgb_passado_uint8).save(f"{pasta}/rgb_passado.png")
    Image.fromarray(rgb_atual_uint8).save(f"{pasta}/rgb_atual.png")
    Image.fromarray(((ndvi_passado + 1) / 2 * 255).astype(np.uint8)).save(f"{pasta}/ndvi_passado.png")
    Image.fromarray(((ndvi_atual + 1) / 2 * 255).astype(np.uint8)).save(f"{pasta}/ndvi_atual.png")
    Image.fromarray((mascara_perda.astype(np.uint8) * 255)).save(f"{pasta}/mascara_perda.png")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title(f"NDVI {data_passada}")
    plt.imshow(ndvi_passado, cmap='YlGn')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"NDVI {data_atual}")
    plt.imshow(ndvi_atual, cmap='YlGn')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Perda Vegetação\n({percentual:.2f}% - {classificacao})")
    plt.imshow(mascara_perda, cmap='Reds')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{pasta}/comparativo.png")
    plt.close()

    resultados.append({
        "Latitude": lat,
        "Longitude": lon,
        "Data_Passada": str(data_passada),
        "Data_Atual": str(data_atual),
        "Perda (%)": round(percentual, 2),
        "Classificação": classificacao
    })

# === SALVAR CSV FINAL ===
df = pd.DataFrame(resultados)
df.to_csv("resultados_ndvi/resultados.csv", index=False)
print("\n✅ Processamento finalizado. Resultados salvos em 'resultados_ndvi/resultados.csv'")
