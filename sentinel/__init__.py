import ee
import folium

# Inicializar o Earth Engine
ee.Initialize(project='deft-smile-462318-v2')  # Substitua pelo seu projeto GEE

# Definir o período de datas (substitua pelas datas desejadas)
start_date = '2022-01-01'  # Data inicial
end_date = '2022-06-19'    # Data final (hoje, 19/06/2025, 12:18 PM -03)

# Definir a geometria de Blumenau (coordenadas aproximadas)
blumenau_area = ee.Geometry.Polygon([
    [
        [-49.1200, -26.9600],  # Canto sudoeste
        [-49.1200, -26.8500],  # Canto noroeste
        [-49.0000, -26.8500],  # Canto nordeste
        [-49.0000, -26.9600],  # Canto sudeste
        [-49.1200, -26.9600]   # Fechar o polígono
    ]
])

# Carregar imagens Sentinel-2
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(blumenau_area) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))

# Verificar se há imagens na coleção
size = collection.size().getInfo()
print(f"Número de imagens na coleção: {size}")

if size == 0:
    print(f"Nenhuma imagem encontrada para o período {start_date} a {end_date}. Tente ajustar o intervalo de datas ou o filtro de nuvens.")
else:
    # Obter a imagem mais recente
    image = collection.sort('system:time_start', False).first().clip(blumenau_area)

    # Obter a data e a porcentagem de nuvens da imagem
    image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    cloud_percentage = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    print(f"Imagem mais recente encontrada: {image_date}")
    print(f"Porcentagem de nuvens: {cloud_percentage}%")

    # Parâmetros de visualização (RGB)
    vis_params = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2']  # Vermelho, Verde, Azul
    }

    # Criar um mapa Folium centrado em Blumenau
    mapa = folium.Map(location=[-26.9050, -49.0600], zoom_start=12)

    # Adicionar a camada de imagem
    map_id_dict = ee.Image(image).getMapId(vis_params)
    folium.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data © <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name='Sentinel-2',
        overlay=True,
        control=True
    ).add_to(mapa)

    # Adicionar controle de camadas e salvar o mapa
    folium.LayerControl().add_to(mapa)
    mapa.save("blumenau_map.html")
    print("✅ Mapa salvo como blumenau_map.html. Abra no navegador.")