from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, SentinelHubCatalog
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from PIL import Image
import os
import pandas as pd
import re
import math

# === CONFIGURAÇÃO DO SENTINEL HUB ===
config = SHConfig()
config.sh_client_id = '69b0e122-2c43-444c-a03a-11f80f0fa3f6'
config.sh_client_secret = 'UzfLRBn4lWFxz9hypPOeWKxV4BW8LYsT'

# === VALORES PADRÃO ===
TAMANHO_JANELA_PADRAO = 0.0050  # ~500m de raio
LIMIAR_VEGETACAO_PADRAO = 0.4
LIMIAR_PERDA_PADRAO = -0.2
# === PARÂMETROS PARA THRESHOLD ADAPTATIVO ===
FACTOR_STD_PADRAO = 1.7  # Fator multiplicador do desvio padrão
USAR_THRESHOLD_ADAPTATIVO = True  # Flag para ativar/desativar threshold adaptativo

hoje = date.today()

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

def calcular_area_hectares(tamanho_janela, latitude):
    """
    Calcula a área aproximada em hectares baseada no tamanho da janela e latitude
    """
    # Conversão aproximada de graus para metros na latitude especificada
    # 1 grau de longitude varia com a latitude: cos(lat) * 111320 metros
    # 1 grau de latitude é aproximadamente 111320 metros

    lat_rad = math.radians(latitude)
    metros_por_grau_lon = math.cos(lat_rad) * 111320
    metros_por_grau_lat = 111320

    # Área do retângulo (2 * tamanho_janela para cada dimensão)
    largura_metros = 2 * tamanho_janela * metros_por_grau_lon
    altura_metros = 2 * tamanho_janela * metros_por_grau_lat

    area_m2 = largura_metros * altura_metros
    area_hectares = area_m2 / 10000  # Converter para hectares

    return round(area_hectares, 2)

def sanitizar_nome_pasta(nome):
    """
    Remove caracteres inválidos para nomes de pasta e limita o tamanho
    """
    # Remover caracteres especiais e substituir por underscore
    nome_limpo = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', nome)
    # Remover espaços extras e substituir por underscore
    nome_limpo = re.sub(r'\s+', '_', nome_limpo.strip())
    # Limitar tamanho para evitar problemas de sistema
    if len(nome_limpo) > 50:
        nome_limpo = nome_limpo[:50]
    return nome_limpo

def parse_data_limite(data_str):
    """
    Converte string de data para objeto date.
    Aceita formatos: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY
    """
    if not data_str or not data_str.strip():
        return None

    data_str = data_str.strip()

    try:
        # Formato ISO (YYYY-MM-DD)
        if '-' in data_str and len(data_str.split('-')[0]) == 4:
            return datetime.strptime(data_str, '%Y-%m-%d').date()

        # Formato brasileiro (DD/MM/YYYY)
        elif '/' in data_str:
            return datetime.strptime(data_str, '%d/%m/%Y').date()

        # Formato DD-MM-YYYY
        elif '-' in data_str:
            return datetime.strptime(data_str, '%d-%m-%Y').date()

        else:
            print(f"Formato de data não reconhecido: {data_str}")
            return None

    except ValueError as e:
        print(f"Erro ao converter data '{data_str}': {e}")
        return None

def ler_coordenadas_arquivo(nome_arquivo):
    """
    Lê coordenadas de arquivo texto no formato:
    label;latitude;longitude;tamanho_janela;limiar_perda;data_limite;factor_std;usar_adaptativo

    Onde tamanho_janela, limiar_perda, data_limite, factor_std e usar_adaptativo são opcionais
    data_limite: data no formato YYYY-MM-DD, DD/MM/YYYY ou DD-MM-YYYY
                Se não fornecida, usa as 2 imagens mais recentes
    factor_std: fator multiplicador do desvio padrão (padrão: 3)
    usar_adaptativo: 1 para usar threshold adaptativo, 0 para usar fixo (padrão: 1)
    """
    coordenadas = []

    if not os.path.exists(nome_arquivo):
        print(f"Arquivo {nome_arquivo} não encontrado!")
        print("Criando arquivo de exemplo...")
        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            f.write("# ARQUIVO DE COORDENADAS PARA MONITORAMENTO DE VEGETAÇÃO\n")
            f.write("#\n")
            f.write("# Formato: label;latitude;longitude;tamanho_janela;limiar_perda;data_limite;factor_std;usar_adaptativo\n")
            f.write("#\n")
            f.write("# PARÂMETROS:\n")
            f.write("# - label: nome/rótulo do local (obrigatório, usado no nome da pasta)\n")
            f.write("# - latitude: coordenada latitude em graus decimais (obrigatório)\n")
            f.write("# - longitude: coordenada longitude em graus decimais (obrigatório)\n")
            f.write("# - tamanho_janela: raio da área em graus (opcional, padrão: 0.0050 ≈ 500m)\n")
            f.write("# - limiar_perda: threshold para detectar perda (opcional, padrão: -0.2)\n")
            f.write("# - data_limite: data limite para busca (opcional, formatos: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY)\n")
            f.write("#   * Se fornecida: compara imagem mais recente antes desta data com a mais recente atual\n")
            f.write("#   * Se não fornecida: compara as 2 imagens mais recentes disponíveis\n")
            f.write("# - factor_std: fator multiplicador do desvio padrão (opcional, padrão: 3)\n")
            f.write("#   * Usado apenas quando usar_adaptativo=1\n")
            f.write("#   * Valores maiores = menos sensível, valores menores = mais sensível\n")
            f.write("# - usar_adaptativo: 1 para threshold adaptativo, 0 para fixo (opcional, padrão: 1)\n")
            f.write("#\n")
            f.write("# NOTAS:\n")
            f.write("# - Linhas iniciadas com # são comentários\n")
            f.write("# - Valores opcionais podem ser deixados em branco\n")
            f.write("# - Labels não devem conter caracteres especiais (<>:\"/\\|?*)\n")
            f.write("# - Tamanho da janela: 0.0050 ≈ 500m, 0.0100 ≈ 1km, 0.0025 ≈ 250m\n")
            f.write("# - Limiar de perda: valores mais negativos detectam perdas maiores\n")
            f.write("#   -0.2 = mais sensível, -0.7 = menos sensível\n")
            f.write("# - Threshold adaptativo: calcula automaticamente o limiar baseado no desvio padrão\n")
            f.write("#   das mudanças de NDVI, tornando a análise mais robusta\n")
            f.write("#\n")
            f.write("# EXEMPLOS:\n")
            f.write("\n")
            f.write("# Fazenda Norte: Threshold adaptativo com fator 3 (padrão)\n")
            f.write("Fazenda_Norte;-26.90567708545667;-49.0556474962708;0.0050;-0.2;;3;1\n")
            f.write("\n")
            f.write("# Área de Preservação: Threshold adaptativo mais sensível (fator 2)\n")
            f.write("Area_Preservacao;-26.96046686848641;-49.145220530125506;;;;;2;1\n")
            f.write("\n")
            f.write("# Mata Ciliar: Threshold fixo tradicional\n")
            f.write("Mata_Ciliar;-26.868825565055406;-49.16984456369647;0.0100;-0.3;;;0\n")
            f.write("\n")
            f.write("# Reflorestamento: Threshold adaptativo menos sensível (fator 4)\n")
            f.write("Reflorestamento;-26.846305514170737;-49.11930082624259;0.0025;-0.7;;4;1\n")
            f.write("\n")
            f.write("# Campo Nativo: Configuração padrão (threshold adaptativo)\n")
            f.write("Campo_Nativo;-26.915;-49.065;;;;;;;1\n")
            f.write("\n")
            f.write("# Adicione suas coordenadas abaixo:\n")
        print(f"Arquivo {nome_arquivo} criado com exemplos. Edite-o com suas coordenadas.")
        return []

    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            for linha_num, linha in enumerate(f, 1):
                linha = linha.strip()
                if not linha or linha.startswith('#'):
                    continue

                partes = linha.split(';')
                if len(partes) < 3:
                    print(f"Linha {linha_num} ignorada (formato inválido): {linha}")
                    continue

                try:
                    label = partes[0].strip()
                    if not label:
                        label = f"Ponto_{linha_num}"

                    lat = float(partes[1])
                    lon = float(partes[2])
                    tamanho_janela = float(partes[3]) if len(partes) > 3 and partes[3] else TAMANHO_JANELA_PADRAO
                    limiar_perda = float(partes[4]) if len(partes) > 4 and partes[4] else LIMIAR_PERDA_PADRAO

                    # Parâmetro: data limite
                    data_limite = None
                    if len(partes) > 5 and partes[5]:
                        data_limite = parse_data_limite(partes[5])

                    # Novos parâmetros para threshold adaptativo
                    factor_std = float(partes[6]) if len(partes) > 6 and partes[6] else FACTOR_STD_PADRAO
                    usar_adaptativo = bool(int(partes[7])) if len(partes) > 7 and partes[7] else USAR_THRESHOLD_ADAPTATIVO

                    coordenadas.append((label, lat, lon, tamanho_janela, limiar_perda, data_limite, factor_std, usar_adaptativo))
                except ValueError as e:
                    print(f"Linha {linha_num} ignorada (erro de conversão): {linha} - {e}")
                    continue

    except Exception as e:
        print(f"Erro ao ler arquivo {nome_arquivo}: {e}")
        return []

    return coordenadas

def buscar_imagem_com_data_limite(catalog, aoi, data_limite):
    """
    Busca uma imagem mais recente antes da data limite e a mais recente disponível
    """
    # Buscar imagem mais recente antes da data limite
    data_inicio_antiga = data_limite - timedelta(days=2*365)  # Buscar até 2 anos antes

    search_antiga = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=aoi,
        time=(data_inicio_antiga, data_limite),
        filter='eo:cloud_cover < 20',
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
        limit=20
    )

    results_antiga = list(search_antiga)

    # Buscar imagem mais recente (últimos 6 meses)
    data_recente_inicio = hoje - timedelta(days=180)
    data_recente_fim = hoje

    search_recente = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=aoi,
        time=(data_recente_inicio, data_recente_fim),
        filter='eo:cloud_cover < 20',
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
        limit=20
    )

    results_recente = list(search_recente)

    if not results_antiga or not results_recente:
        return None, None

    # Pegar a melhor imagem antes da data limite (ordenar por data descendente, depois por nuvem)
    results_antiga.sort(key=lambda x: (-datetime.fromisoformat(x["properties"]["datetime"].replace("Z", "+00:00")).timestamp(),
                                       x["properties"]["eo:cloud_cover"]))
    melhor_antiga = results_antiga[0]
    dt_antiga = datetime.fromisoformat(melhor_antiga["properties"]["datetime"].replace("Z", "+00:00"))
    data_antiga = (dt_antiga.date(), melhor_antiga["properties"]["eo:cloud_cover"])

    # Pegar a imagem mais recente (ordenar por data, depois por nuvem)
    results_recente.sort(key=lambda x: (-datetime.fromisoformat(x["properties"]["datetime"].replace("Z", "+00:00")).timestamp(),
                                        x["properties"]["eo:cloud_cover"]))
    melhor_recente = results_recente[0]
    dt_recente = datetime.fromisoformat(melhor_recente["properties"]["datetime"].replace("Z", "+00:00"))
    data_recente = (dt_recente.date(), melhor_recente["properties"]["eo:cloud_cover"])

    return data_recente, data_antiga

def buscar_datas_validas(catalog, aoi, data_limite_recente, data_limite_antiga, num_imagens=2):
    """
    Busca as duas imagens mais recentes com baixa cobertura de nuvem
    """
    search_iterator = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=aoi,
        time=(data_limite_antiga, data_limite_recente),
        filter='eo:cloud_cover < 20',
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
        limit=50  # Buscar mais resultados para ter opções
    )

    results = list(search_iterator)
    if len(results) < 2:
        return None, None

    # Ordenar por data (mais recente primeiro) e depois por cobertura de nuvem
    results.sort(key=lambda x: (x["properties"]["datetime"], x["properties"]["eo:cloud_cover"]), reverse=True)

    datas = []
    for result in results[:num_imagens]:
        dt_str = result["properties"]["datetime"]
        dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        cobertura_nuvem = result["properties"]["eo:cloud_cover"]
        datas.append((dt_obj.date(), cobertura_nuvem))

    if len(datas) >= 2:
        return datas[0], datas[1]  # (data_mais_recente, cobertura), (data_mais_antiga, cobertura)

    return None, None

def requisicao_rgb(data, aoi, width, height):
    global config
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
    global config
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

def classificar_perda(percentual):
    if percentual < 0.5:
        return "Inalterado"
    elif percentual < 1:
        return "Alerta"
    elif percentual < 5:
        return "Grave"
    else:
        return "Gravissimo"

def calcular_threshold_adaptativo(delta_ndvi, mascara_veg, factor_std):
    """
    Calcula o threshold adaptativo baseado no desvio padrão das mudanças de NDVI
    nas áreas vegetadas
    """
    if not np.any(mascara_veg):
        return -0.2  # Fallback para valor padrão se não houver vegetação

    # Calcular desvio padrão apenas nas áreas vegetadas
    std_dev = np.std(delta_ndvi[mascara_veg])

    # Threshold adaptativo: -factor_std * desvio_padrão
    threshold_adaptativo = -factor_std * std_dev

    return threshold_adaptativo

def main():
    global config

    # Nome do arquivo de coordenadas
    arquivo_coordenadas = "coordenadas.txt"

    print("Monitor de Vegetacao - Versao com Threshold Adaptativo")
    print("=" * 60)

    # Ler coordenadas do arquivo
    coordenadas = ler_coordenadas_arquivo(arquivo_coordenadas)

    if not coordenadas:
        print("Nenhuma coordenada valida encontrada. Programa encerrado.")
        return

    print(f"{len(coordenadas)} coordenada(s) carregada(s)")

    # === ARMAZENAR RESULTADOS ===
    resultados = []

    # === PROCESSAR TODAS AS COORDENADAS ===
    for idx, (label, lat, lon, tamanho_janela, limiar_perda, data_limite, factor_std, usar_adaptativo) in enumerate(coordenadas):
        print(f"\nProcessando {label} (ponto {idx+1}/{len(coordenadas)}): ({lat:.5f}, {lon:.5f})")
        print(f"   Janela: ±{tamanho_janela:.4f}° | Limiar fixo: {limiar_perda}")
        print(f"   Threshold: {'Adaptativo (fator=' + str(factor_std) + ')' if usar_adaptativo else 'Fixo'}")

        if data_limite:
            modo_comparacao = f"Data limite: {data_limite} vs atual"
        else:
            modo_comparacao = "2 imagens mais recentes"
        print(f"   Modo: {modo_comparacao}")

        # Calcular área em hectares
        area_hectares = calcular_area_hectares(tamanho_janela, lat)

        # Criar área de interesse
        aoi = BBox([lon - tamanho_janela, lat - tamanho_janela,
                    lon + tamanho_janela, lat + tamanho_janela], crs=CRS.WGS84)
        width, height = bbox_to_dimensions(aoi, 10)

        # Buscar imagens baseado na configuração
        catalog = SentinelHubCatalog(config=config)

        if data_limite:
            # Buscar uma imagem antes da data limite e uma recente
            resultado_recente, resultado_antigo = buscar_imagem_com_data_limite(catalog, aoi, data_limite)
        else:
            # Buscar as duas imagens mais recentes nos últimos 2 anos
            data_limite_recente = hoje
            data_limite_antiga = hoje - timedelta(days=2 * 365)

            resultado_recente, resultado_antigo = buscar_datas_validas(
                catalog, aoi, data_limite_recente, data_limite_antiga
            )

        if not resultado_recente or not resultado_antigo:
            erro_msg = "Imagens adequadas nao encontradas"
            if data_limite:
                erro_msg += f" (sem imagens antes de {data_limite} ou recentes)"
            print(f"  {erro_msg}")
            resultados.append({
                "Label": label,
                "Ponto": f"{idx+1}",
                "Latitude": lat,
                "Longitude": lon,
                "Area_Hectares": area_hectares,
                "Tamanho_Janela": tamanho_janela,
                "Limiar_Perda_Fixo": limiar_perda,
                "Factor_Std": factor_std,
                "Usar_Adaptativo": usar_adaptativo,
                "Threshold_Usado": "N/A",
                "Modo_Comparacao": modo_comparacao,
                "Data_Limite": str(data_limite) if data_limite else "N/A",
                "Data_Solicitada": str(hoje),
                "Data_Imagem_Recente": "N/A",
                "Cobertura_Nuvem_Recente (%)": "N/A",
                "Data_Imagem_Antiga": "N/A",
                "Cobertura_Nuvem_Antiga (%)": "N/A",
                "Perda_Vegetacao (%)": "N/A",
                "Classificacao": "Sem dados",
                "Pixels_Vegetacao": "N/A",
                "Pixels_Perda": "N/A",
                "Status": f"Erro - {erro_msg.lower()}"
            })
            continue

        data_recente, cobertura_recente = resultado_recente
        data_antiga, cobertura_antiga = resultado_antigo

        periodo_comparacao = f"{data_antiga} -> {data_recente}"
        dias_diferenca = (data_recente - data_antiga).days
        print(f"  Comparacao: {periodo_comparacao} ({dias_diferenca} dias)")
        print(f"  Nuvens: {cobertura_antiga:.1f}% -> {cobertura_recente:.1f}%")

        try:
            # Fazer requisições
            ndvi_recente = requisicao_ndvi(data_recente, aoi, width, height).get_data()[0].squeeze()
            ndvi_antigo = requisicao_ndvi(data_antiga, aoi, width, height).get_data()[0].squeeze()

            rgb_recente = requisicao_rgb(data_recente, aoi, width, height).get_data()[0]
            rgb_antigo = requisicao_rgb(data_antiga, aoi, width, height).get_data()[0]

            # Processar imagens RGB
            rgb_recente_uint8 = np.clip(rgb_recente / np.max(rgb_recente) * 255, 0, 255).astype(np.uint8)
            rgb_antigo_uint8 = np.clip(rgb_antigo / np.max(rgb_antigo) * 255, 0, 255).astype(np.uint8)

            # Calcular diferenças
            delta_ndvi = ndvi_recente - ndvi_antigo
            mascara_veg = ndvi_antigo > LIMIAR_VEGETACAO_PADRAO

            # === IMPLEMENTAÇÃO DO THRESHOLD ADAPTATIVO ===
            if usar_adaptativo:
                threshold_usado = calcular_threshold_adaptativo(delta_ndvi, mascara_veg, factor_std)
                print(f"  Threshold adaptativo calculado: {threshold_usado:.4f}")
            else:
                threshold_usado = limiar_perda
                print(f"  Usando threshold fixo: {threshold_usado}")

            # Aplicar threshold (adaptativo ou fixo)
            mascara_perda = (delta_ndvi < threshold_usado) & mascara_veg

            pixels_vegetacao = np.count_nonzero(mascara_veg)
            pixels_perda = np.count_nonzero(mascara_perda)
            percentual_perda = (pixels_perda / pixels_vegetacao) * 100 if pixels_vegetacao else 0
            classificacao = classificar_perda(percentual_perda)

            print(f"  Perda de vegetacao: {percentual_perda:.2f}% ({classificacao})")

            # Criar nome da pasta com label
            label_limpo = sanitizar_nome_pasta(label)
            pasta = f"resultados_ndvi/ponto_{idx + 1}_{label_limpo}"
            os.makedirs(pasta, exist_ok=True)

            # Salvar imagens básicas
            Image.fromarray(rgb_antigo_uint8).save(f"{pasta}/rgb_antigo.png")
            Image.fromarray(rgb_recente_uint8).save(f"{pasta}/rgb_recente.png")
            Image.fromarray(((ndvi_antigo + 1) / 2 * 255).astype(np.uint8)).save(f"{pasta}/ndvi_antigo.png")
            Image.fromarray(((ndvi_recente + 1) / 2 * 255).astype(np.uint8)).save(f"{pasta}/ndvi_recente.png")
            Image.fromarray((mascara_perda.astype(np.uint8) * 255)).save(f"{pasta}/mascara_perda.png")

            # Criar imagem com sobreposição da máscara de perda na imagem RGB atual
            overlay_rgb = rgb_recente_uint8.copy()
            mascara_rgb = np.zeros_like(overlay_rgb)
            mascara_rgb[..., 0] = 255  # vermelho
            alpha = 0.5
            overlay_rgb[mascara_perda] = (
                    (1 - alpha) * overlay_rgb[mascara_perda] + alpha * mascara_rgb[mascara_perda]
            ).astype(np.uint8)

            # Gerar imagem com sobreposição salva
            Image.fromarray(overlay_rgb).save(f"{pasta}/rgb_com_mascara.png")

            plt.figure(figsize=(18, 5))

            # Configuração dos plots
            plot_configs = [
                {'img': rgb_antigo_uint8, 'title': f"Imagem RGB Antigo\n{data_antiga}", 'cmap': 'YlGn'},
                {'img': rgb_recente_uint8, 'title': f"Imagem RGB Recente\n{data_recente}", 'cmap': 'YlGn'},
                {'img': delta_ndvi, 'title': "Diferença NDVI\n(Recente - Antigo)", 'cmap': 'RdYlGn'},
                {'img': overlay_rgb,
                 'title': f"RGB Recente + Máscara de Perda\n{percentual_perda:.2f}% - {classificacao}", 'cmap': None}
            ]

            for i, config_plot in enumerate(plot_configs, 1):
                plt.subplot(1, 4, i)
                if config_plot['cmap']:
                    img = plt.imshow(config_plot['img'], cmap=config_plot['cmap'], vmin=-1 if i == 3 else 0,
                                     vmax=1 if i == 3 else 255)
                else:
                    img = plt.imshow(config_plot['img'])  # para RGB com overlay
                plt.axis('off')
                plt.title(config_plot['title'], y=-0.18, pad=10)

                if i == 3:
                    plt.colorbar(img, shrink=0.8, pad=0.02)

            # Legenda personalizada para a sobreposição
            legend_elements = [
                Patch(facecolor='red', edgecolor='black', label='Perda de vegetação')
            ]
            plt.legend(
                handles=legend_elements,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.25),
                frameon=True,
                ncol=1,
                title="Legenda:"
            )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25, wspace=0.2)
            plt.savefig(f"{pasta}/analise_comparativa.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Adicionar aos resultados
            resultados.append({
                "Label": label,
                "Ponto": f"{idx+1}",
                "Latitude": lat,
                "Longitude": lon,
                "Area_Hectares": area_hectares,
                "Tamanho_Janela": tamanho_janela,
                "Limiar_Perda_Fixo": limiar_perda,
                "Factor_Std": factor_std,
                "Usar_Adaptativo": usar_adaptativo,
                "Threshold_Usado": round(threshold_usado, 4),
                "Modo_Comparacao": modo_comparacao,
                "Data_Limite": str(data_limite) if data_limite else "N/A",
                "Data_Solicitada": str(hoje),
                "Data_Imagem_Recente": str(data_recente),
                "Cobertura_Nuvem_Recente (%)": round(cobertura_recente, 1),
                "Data_Imagem_Antiga": str(data_antiga),
                "Cobertura_Nuvem_Antiga (%)": round(cobertura_antiga, 1),
                "Perda_Vegetacao (%)": round(percentual_perda, 2),
                "Classificacao": classificacao,
                "Pixels_Vegetacao": int(pixels_vegetacao),
                "Pixels_Perda": int(pixels_perda),
                "Status": "Sucesso"
            })

        except Exception as e:
            print(f"  Erro no processamento: {e}")
            resultados.append({
                "Label": label,
                "Ponto": f"{idx+1}",
                "Latitude": lat,
                "Longitude": lon,
                "Area_Hectares": area_hectares,
                "Tamanho_Janela": tamanho_janela,
                "Limiar_Perda_Fixo": limiar_perda,
                "Factor_Std": factor_std,
                "Usar_Adaptativo": usar_adaptativo,
                "Threshold_Usado": "N/A",
                "Modo_Comparacao": modo_comparacao,
                "Data_Limite": str(data_limite) if data_limite else "N/A",
                "Data_Solicitada": str(hoje),
                "Data_Imagem_Recente": "N/A",
                "Cobertura_Nuvem_Recente (%)": "N/A",
                "Data_Imagem_Antiga": "N/A",
                "Cobertura_Nuvem_Antiga (%)": "N/A",
                "Perda_Vegetacao (%)": "N/A",
                "Classificacao": "Erro",
                "Pixels_Vegetacao": "N/A",
                "Pixels_Perda": "N/A",
                "Status": f"Erro: {str(e)}"
            })

    # === SALVAR RESULTADOS FINAIS ===
    if resultados:
        os.makedirs("resultados_ndvi", exist_ok=True)
        df = pd.DataFrame(resultados)

        # Salvar apenas um CSV com todos os resultados
        nome_csv = "resultados_ndvi/analise_vegetacao_completa.csv"
        df.to_csv(nome_csv, index=False, encoding='utf-8-sig')

        print(f"\nProcessamento finalizado!")
        print(f"Resultados salvos em: {nome_csv}")

        # Mostrar resumo
        sucessos = len([r for r in resultados if r["Status"] == "Sucesso"])
        print(f"\nResumo: {sucessos}/{len(resultados)} pontos processados com sucesso")

        if sucessos > 0:
            df_sucesso = df[df["Status"] == "Sucesso"]
            perdas = df_sucesso["Perda_Vegetacao (%)"].astype(float)
            print(f"Perda media de vegetacao: {perdas.mean():.2f}%")
            print(f"Maior perda: {perdas.max():.2f}%")

            # Mostrar resultados por label
            print(f"\nResultados por area:")
            for _, row in df_sucesso.iterrows():
                print(f"  {row['Label']}: {row['Perda_Vegetacao (%)']}% ({row['Classificacao']})")

    else:
        print("\nNenhum resultado foi gerado.")

if __name__ == "__main__":
    main()