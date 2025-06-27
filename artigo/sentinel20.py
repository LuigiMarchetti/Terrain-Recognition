from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, \
    SentinelHubCatalog
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from PIL import Image
import os
import pandas as pd
import re
import math


config = SHConfig()
config.sh_client_id = '69b0e122-2c43-444c-a03a-11f80f0fa3f6'
config.sh_client_secret = 'UzfLRBn4lWFxz9hypPOeWKxV4BW8LYsT'

# === VALORES PADRÃO ===
TAMANHO_JANELA_PADRAO = 0.0050
LIMIAR_VEGETACAO_PADRAO = 0.4
LIMIAR_PERDA_PADRAO = -0.2
FACTOR_N_PADRAO = 1.5  # Padrão agora é 1.5, alinhado com o artigo
MODO_THRESHOLD_PADRAO = 2  # Padrão agora é 2 (método do artigo)

hoje = date.today()

# === EVALSCRIPTS ===
evalscript_rgb = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3, sampleType: "AUTO" }
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
  // Mascara nuvem, sombra, neve, etc. Retorna um valor nulo que o Python lerá como NaN
  if ([3, 6, 8, 9, 10, 11].includes(sample.SCL)) {
    return [NaN];
  }
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi];
}
"""


def calcular_area_hectares(tamanho_janela, latitude):
    lat_rad = math.radians(latitude)
    metros_por_grau_lon = math.cos(lat_rad) * 111320
    metros_por_grau_lat = 111320
    largura_metros = 2 * tamanho_janela * metros_por_grau_lon
    altura_metros = 2 * tamanho_janela * metros_por_grau_lat
    area_m2 = largura_metros * altura_metros
    return round(area_m2 / 10000, 2)


def sanitizar_nome_pasta(nome):
    nome_limpo = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', nome)
    nome_limpo = re.sub(r'\s+', '_', nome_limpo.strip())
    return nome_limpo[:50]


def parse_data_limite(data_str):
    if not data_str or not data_str.strip():
        return None
    data_str = data_str.strip()
    try:
        if '-' in data_str and len(data_str.split('-')[0]) == 4:
            return datetime.strptime(data_str, '%Y-%m-%d').date()
        elif '/' in data_str:
            return datetime.strptime(data_str, '%d/%m/%Y').date()
        elif '-' in data_str:
            return datetime.strptime(data_str, '%d-%m-%Y').date()
        return None
    except ValueError as e:
        print(f"Erro ao converter data '{data_str}': {e}")
        return None


def ler_coordenadas_arquivo(nome_arquivo):
    coordenadas = []
    if not os.path.exists(nome_arquivo):
        print(f"Arquivo {nome_arquivo} não encontrado! Criando arquivo de exemplo...")
        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            f.write("# ARQUIVO DE COORDENADAS PARA MONITORAMENTO DE VEGETAÇÃO\n")
            f.write("#\n")
            f.write(
                "# Formato: label;latitude;longitude;tamanho_janela;limiar_perda_fixo;data_limite;fator_n;modo_threshold\n")
            f.write("#\n")
            f.write("# PARÂMETROS:\n")
            f.write("# - label, latitude, longitude: obrigatórios.\n")
            f.write("# - tamanho_janela: raio da área em graus (opcional, padrão: 0.0050 ≈ 500m).\n")
            f.write("# - limiar_perda_fixo: threshold para o modo fixo (opcional, padrão: -0.2).\n")
            f.write("# - data_limite: data para busca (opcional, formatos: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY).\n")
            f.write("# - fator_n: fator multiplicador do desvio padrão (opcional, padrão: 1.5).\n")
            f.write("# - modo_threshold: seleciona o método de cálculo do limiar (opcional, padrão: 2).\n")
            f.write("#   -> 0 = Fixo (usa o valor de 'limiar_perda_fixo').\n")
            f.write("#   -> 1 = Adaptativo (baseado no desvio padrão da vegetação estável, como no código original).\n")
            f.write("#   -> 2 = Artigo (baseado na média e desvio padrão da imagem de diferença inteira).\n")
            f.write("#\n")
            f.write("# EXEMPLOS:\n")
            f.write("# Fazenda Teste: Usando método do artigo com n=1.5 (padrão do artigo)\n")
            f.write("Fazenda_Teste;-26.9056;-49.0556;;;;1.5;2\n")
            f.write("\n")
            f.write("# Área de Preservação: Usando método adaptativo original (vegetação) com fator 3\n")
            f.write("Area_Preservacao;-26.9604;-49.1452;;;;3;1\n")
            f.write("\n")
            f.write("# Mata Ciliar: Usando método de threshold fixo -0.3\n")
            f.write("Mata_Ciliar;-26.8688;-49.1698;0.0100;-0.3;;;0\n")
        print(f"Arquivo {nome_arquivo} criado com exemplos. Edite-o com suas coordenadas.")
        return []

    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            for linha_num, linha in enumerate(f, 1):
                linha = linha.strip()
                if not linha or linha.startswith('#'):
                    continue
                partes = [p.strip() for p in linha.split(';')]
                if len(partes) < 3:
                    print(f"Linha {linha_num} ignorada (formato inválido): {linha}")
                    continue
                try:
                    label = partes[0] if partes[0] else f"Ponto_{linha_num}"
                    lat = float(partes[1])
                    lon = float(partes[2])
                    tamanho_janela = float(partes[3]) if len(partes) > 3 and partes[3] else TAMANHO_JANELA_PADRAO
                    limiar_perda_fixo = float(partes[4]) if len(partes) > 4 and partes[4] else LIMIAR_PERDA_PADRAO
                    data_limite = parse_data_limite(partes[5]) if len(partes) > 5 and partes[5] else None
                    factor_n = float(partes[6]) if len(partes) > 6 and partes[6] else FACTOR_N_PADRAO
                    modo_threshold = int(partes[7]) if len(partes) > 7 and partes[7] else MODO_THRESHOLD_PADRAO
                    coordenadas.append(
                        (label, lat, lon, tamanho_janela, limiar_perda_fixo, data_limite, factor_n, modo_threshold))
                except (ValueError, IndexError) as e:
                    print(f"Linha {linha_num} ignorada (erro de conversão de valor): {linha} - {e}")
    except Exception as e:
        print(f"Erro fatal ao ler arquivo {nome_arquivo}: {e}")
    return coordenadas


def buscar_datas_validas(catalog, aoi, data_limite_recente, data_limite_antiga, num_imagens=2):
    search_iterator = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=aoi,
        time=(data_limite_antiga, data_limite_recente),
        filter='eo:cloud_cover < 20',
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
        limit=50
    )
    results = list(search_iterator)
    if len(results) < 2:
        return None, None
    results.sort(key=lambda x: x["properties"]["datetime"], reverse=True)

    # Simplesmente pega os dois mais recentes com nuvem < 20
    melhor_recente_info = results[0]
    melhor_antiga_info = results[1]

    dt_recente = datetime.fromisoformat(melhor_recente_info["properties"]["datetime"].replace("Z", "+00:00"))
    dt_antiga = datetime.fromisoformat(melhor_antiga_info["properties"]["datetime"].replace("Z", "+00:00"))

    data_recente = (dt_recente.date(), melhor_recente_info["properties"]["eo:cloud_cover"])
    data_antiga = (dt_antiga.date(), melhor_antiga_info["properties"]["eo:cloud_cover"])

    return data_recente, data_antiga


def buscar_imagem_com_data_limite(catalog, aoi, data_limite):
    data_inicio_antiga = data_limite - timedelta(days=2 * 365)
    search_antiga = catalog.search(
        collection=DataCollection.SENTINEL2_L2A, bbox=aoi, time=(data_inicio_antiga, data_limite),
        filter='eo:cloud_cover < 20', fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
        limit=20
    )
    results_antiga = list(search_antiga)

    data_recente_inicio = hoje - timedelta(days=180)
    search_recente = catalog.search(
        collection=DataCollection.SENTINEL2_L2A, bbox=aoi, time=(data_recente_inicio, hoje),
        filter='eo:cloud_cover < 20', fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
        limit=20
    )
    results_recente = list(search_recente)

    if not results_antiga or not results_recente:
        return None, None

    results_antiga.sort(key=lambda x: (
    -datetime.fromisoformat(x["properties"]["datetime"].replace("Z", "+00:00")).timestamp(),
    x["properties"]["eo:cloud_cover"]))
    melhor_antiga = results_antiga[0]
    dt_antiga = datetime.fromisoformat(melhor_antiga["properties"]["datetime"].replace("Z", "+00:00"))
    data_antiga = (dt_antiga.date(), melhor_antiga["properties"]["eo:cloud_cover"])

    results_recente.sort(key=lambda x: (
    -datetime.fromisoformat(x["properties"]["datetime"].replace("Z", "+00:00")).timestamp(),
    x["properties"]["eo:cloud_cover"]))
    melhor_recente = results_recente[0]
    dt_recente = datetime.fromisoformat(melhor_recente["properties"]["datetime"].replace("Z", "+00:00"))
    data_recente = (dt_recente.date(), melhor_recente["properties"]["eo:cloud_cover"])
    return data_recente, data_antiga


def requisicao_rgb(data, aoi, width, height):
    return SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A, time_interval=(data, data)
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi, size=(width, height), config=config
    )


def requisicao_ndvi(data, aoi, width, height):
    return SentinelHubRequest(
        evalscript=evalscript_ndvi,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A, time_interval=(data, data)
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi, size=(width, height), config=config
    )


def classificar_perda(percentual):
    if percentual < 0.5:
        return "Inalterado"
    elif percentual < 1:
        return "Alerta"
    elif percentual < 5:
        return "Grave"
    else:
        return "Gravíssimo"


def calcular_threshold_adaptativo_vegetacao(delta_ndvi, mascara_veg, factor_std):
    if not np.any(mascara_veg):
        return LIMIAR_PERDA_PADRAO
    std_dev = np.nanstd(delta_ndvi[mascara_veg])
    threshold = -factor_std * std_dev
    return threshold


def calcular_threshold_artigo(delta_ndvi_validos, n_factor):
    media = np.nanmean(delta_ndvi_validos)
    std_dev = np.nanstd(delta_ndvi_validos)
    threshold = media - (n_factor * std_dev)
    return threshold


def main():
    arquivo_coordenadas = "coordenadas.txt"
    print("Monitor de Vegetação - v3.0 com Múltiplos Métodos de Threshold")
    print("=" * 60)
    coordenadas = ler_coordenadas_arquivo(arquivo_coordenadas)
    if not coordenadas:
        print("\nNenhuma coordenada válida encontrada. Edite o arquivo 'coordenadas.txt' e tente novamente.")
        return

    print(f"\n{len(coordenadas)} coordenada(s) carregada(s) para processamento.")
    resultados = []

    for idx, (label, lat, lon, tamanho_janela, limiar_perda_fixo, data_limite, factor_n, modo_threshold) in enumerate(
            coordenadas):
        print(f"\n[{idx + 1}/{len(coordenadas)}] Processando: {label} ({lat:.5f}, {lon:.5f})")

        modo_str = {0: "Fixo", 1: "Adaptativo (Vegetação)", 2: "Adaptativo (Artigo)"}.get(modo_threshold,
                                                                                          "Desconhecido")
        print(f"  Modo de Threshold: {modo_str}")

        aoi = BBox([lon - tamanho_janela, lat - tamanho_janela, lon + tamanho_janela, lat + tamanho_janela],
                   crs=CRS.WGS84)
        width, height = bbox_to_dimensions(aoi, 10)
        area_hectares = calcular_area_hectares(tamanho_janela, lat)
        catalog = SentinelHubCatalog(config=config)

        if data_limite:
            resultado_recente, resultado_antigo = buscar_imagem_com_data_limite(catalog, aoi, data_limite)
        else:
            resultado_recente, resultado_antigo = buscar_datas_validas(catalog, aoi, hoje,
                                                                       hoje - timedelta(days=2 * 365))

        if not resultado_recente or not resultado_antigo:
            print(f"  Erro: Imagens adequadas não encontradas para o período.")
            # Adicionar lógica para registrar erro no dataframe de resultados
            continue

        data_recente, cobertura_recente = resultado_recente
        data_antiga, cobertura_antiga = resultado_antigo

        print(f"  Comparando: {data_antiga} -> {data_recente} ({(data_recente - data_antiga).days} dias)")
        print(f"  Cobertura de Nuvens: {cobertura_antiga:.1f}% -> {cobertura_recente:.1f}%")

        try:
            print("  Baixando imagens NDVI e RGB...")
            ndvi_recente_raw = requisicao_ndvi(data_recente, aoi, width, height).get_data()[0].squeeze()
            ndvi_antigo_raw = requisicao_ndvi(data_antiga, aoi, width, height).get_data()[0].squeeze()

            valid_data_mask = ~np.isnan(ndvi_recente_raw) & ~np.isnan(ndvi_antigo_raw)
            if not np.any(valid_data_mask):
                print("  Erro: Não há pixels válidos (sem nuvens) em comum entre as duas imagens.")
                continue

            ndvi_recente = np.nan_to_num(ndvi_recente_raw, nan=0.0)
            ndvi_antigo = np.nan_to_num(ndvi_antigo_raw, nan=0.0)

            delta_ndvi = ndvi_recente - ndvi_antigo
            delta_ndvi_validos = delta_ndvi[valid_data_mask]

            mascara_veg = (ndvi_antigo > LIMIAR_VEGETACAO_PADRAO) & valid_data_mask

            if modo_threshold == 1:
                threshold_usado = calcular_threshold_adaptativo_vegetacao(delta_ndvi, mascara_veg, factor_n)
                print(f"  Threshold Adaptativo (Vegetação) calculado: {threshold_usado:.4f}")
            elif modo_threshold == 2:
                threshold_usado = calcular_threshold_artigo(delta_ndvi_validos, factor_n)
                print(f"  Threshold do Artigo (Média - n*σ) calculado: {threshold_usado:.4f}")
            else:  # modo_threshold == 0
                threshold_usado = limiar_perda_fixo
                print(f"  Usando Threshold Fixo: {threshold_usado}")

            mascara_perda = (delta_ndvi < threshold_usado) & mascara_veg

            pixels_vegetacao = np.count_nonzero(mascara_veg)
            pixels_perda = np.count_nonzero(mascara_perda)
            percentual_perda = (pixels_perda / pixels_vegetacao) * 100 if pixels_vegetacao > 0 else 0
            classificacao = classificar_perda(percentual_perda)

            print(f"  Perda de vegetação detectada: {percentual_perda:.2f}% ({classificacao})")

            print("  Gerando e salvando arquivos de imagem...")
            rgb_recente_img = requisicao_rgb(data_recente, aoi, width, height).get_data()[0]
            rgb_antigo_img = requisicao_rgb(data_antiga, aoi, width, height).get_data()[0]

            label_limpo = sanitizar_nome_pasta(label)
            pasta = f"resultados_ndvi/ponto_{idx + 1}_{label_limpo}"
            os.makedirs(pasta, exist_ok=True)

            Image.fromarray(rgb_antigo_img).save(f"{pasta}/rgb_antigo.png")
            Image.fromarray(rgb_recente_img).save(f"{pasta}/rgb_recente.png")
            Image.fromarray(((ndvi_antigo + 1) / 2 * 255).astype(np.uint8)).save(f"{pasta}/ndvi_antigo.png")
            Image.fromarray(((ndvi_recente + 1) / 2 * 255).astype(np.uint8)).save(f"{pasta}/ndvi_recente.png")
            Image.fromarray(mascara_perda.astype(np.uint8) * 255).save(f"{pasta}/mascara_perda.png")

            # Geração do plot comparativo
            fig, axs = plt.subplots(1, 4, figsize=(22, 6))
            fig.suptitle(f"Análise Comparativa para: {label}\n{data_antiga} vs {data_recente}", fontsize=16)

            axs[0].imshow(rgb_antigo_img)
            axs[0].set_title(f"RGB Antigo ({data_antiga})")
            axs[0].axis('off')

            axs[1].imshow(rgb_recente_img)
            axs[1].set_title(f"RGB Recente ({data_recente})")
            axs[1].axis('off')

            im = axs[2].imshow(delta_ndvi, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
            axs[2].set_title("Diferença NDVI (Recente - Antigo)")
            axs[2].axis('off')
            fig.colorbar(im, ax=axs[2], shrink=0.8, label="Variação NDVI")

            # Visualização da perda sobre a imagem recente
            visualizacao_final = rgb_recente_img.copy()
            visualizacao_final[mascara_perda] = [255, 0, 0]  # Vermelho
            axs[3].imshow(visualizacao_final)
            axs[3].set_title(f"Perda Detectada ({percentual_perda:.2f}%)")
            axs[3].axis('off')

            # Salvar imagem de visualização final separadamente
            Image.fromarray(visualizacao_final).save(f"{pasta}/resultado_visual_perda.png")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"{pasta}/analise_comparativa.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Adicionar resultados ao dataframe
            resultados.append({
                "Label": label, "Ponto": f"{idx + 1}", "Latitude": lat, "Longitude": lon,
                "Area_Hectares": area_hectares,
                "Tamanho_Janela": tamanho_janela, "Limiar_Perda_Fixo": limiar_perda_fixo, "Factor_N": factor_n,
                "Modo_Threshold": modo_str, "Threshold_Usado": round(threshold_usado, 4),
                "Data_Limite": str(data_limite) if data_limite else "N/A", "Data_Solicitada": str(hoje),
                "Data_Imagem_Recente": str(data_recente), "Cobertura_Nuvem_Recente (%)": round(cobertura_recente, 1),
                "Data_Imagem_Antiga": str(data_antiga), "Cobertura_Nuvem_Antiga (%)": round(cobertura_antiga, 1),
                "Perda_Vegetacao (%)": round(percentual_perda, 2), "Classificacao": classificacao,
                "Pixels_Vegetacao": int(pixels_vegetacao), "Pixels_Perda": int(pixels_perda),
                "Status": "Sucesso"
            })

        except Exception as e:
            print(f"  ERRO INESPERADO no processamento de '{label}': {e}")
            # Adicionar lógica para registrar erro no dataframe
            continue

    if resultados:
        os.makedirs("resultados_ndvi", exist_ok=True)
        df = pd.DataFrame(resultados)
        nome_csv = "resultados_ndvi/analise_vegetacao_completa.csv"
        df.to_csv(nome_csv, index=False, encoding='utf-8-sig')
        print(f"\n\nProcessamento finalizado! Resultados consolidados salvos em: {nome_csv}")
    else:
        print("\nNenhum resultado foi gerado.")


if __name__ == "__main__":
    main()