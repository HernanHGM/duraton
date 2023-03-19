# %% IMPORTO LIBRERÍAS
import geopandas as gpd
import time
import bokeh
import pandas as pd
import matplotlib.pyplot as plt

import webbrowser
from shapely.geometry import Point, Polygon 
# %% IMPORTO DATOS
file = 'E:\\trabajo_pajaros\\geolocalización\\data1.csv'
df = pd.read_csv(file)

# %% TRANSFORMO A GEOPANDAS
geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)#, crs = 'epsg:4326').to_crs(epsg=3857)

crs = gdf.crs

# Imprimimos el sistema de referencia
print(crs)
 
# %% ELEVACION TERRENO
# import geopandas as gpd
# import time
# file = r'E:/trabajo_pajaros/geolocalización/es_1km.shp'

# start = time.process_time_ns()/(60*10**9)
# a = gpd.read_file(file)

# print('Tiempo transcurrido = ', time.process_time_ns()/(60*10**9) - start, ' segundos')   
# %% def circular_coordinates()

def circular_coordinates(longitude, latitude, r_plane, n):
    import math
    r_deg = 360*r_plane/(6371*2*math.pi)
    x_center = longitude
    y_center = latitude
    coordinates_list = []
    for i in range (1, n+1):
        angle = 2 * math.pi * i/n
        x = x_center + (r_deg * math.cos(angle))
        y = y_center + (r_deg * math.sin(angle))
        coordinates_list.append((x,y))
    return coordinates_list


# %% POLIGONO DE LAS COORDENADAS DE LA ANTENA

longitude, latitude = 40.827332, -4.288363
coordinates_list = circular_coordinates(longitude, latitude, 0.1, 50)

lat_point_list, lon_point_list = zip(*coordinates_list)
polygon_geom = Polygon(zip(lon_point_list, lat_point_list))

polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])  
polygon['name'] = ['antena1']


# %% CLUSTER CON FOLIUM
import folium
from bokeh.tile_providers import get_provider
from folium.plugins import MarkerCluster
 

# Creamos el mapa con Folium
# Añadimos la capa de mapas proporcionada por Bokeh
# m = folium.Map(location=[40.827332,	-4.288363], tiles="Stamen Terrain", zoom_start=12)
m = folium.Map(location=[40.827332, -4.288363], zoom_start=12)
# tile_provider = get_provider('ESRI_IMAGERY')
# folium.TileLayer(tiles=tile_provider.url, attr='Esri, USGS').add_to(m)

# create a marker cluster called "Public toilet cluster"
marker_cluster = MarkerCluster().add_to(m)

for _, r in polygon.iterrows():
    # Without simplifying the representation of each borough,
    # the map might not be displayed
    sim_geo = gpd.GeoSeries(r['geometry'])
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j,
                           style_function=lambda x: {'fillColor': 'orange'})
    folium.Popup(r['name']).add_to(geo_j)
    geo_j.add_to(m)
    
for _, r in gdf.iterrows():
    popup = 'Add <b>test</b>'
    
    # Without simplifying the representation of each borough,
    # the map might not be displayed
    sim_geo = gpd.GeoSeries(r['geometry'])

    geo_j = folium.GeoJson(data=sim_geo.to_json())

    geo_j.add_to(marker_cluster)


file_map = "E:\\trabajo_pajaros\\geolocalización\\map.html"
m.save(file_map)

webbrowser.open(file_map)

# %%

m = folium.Map(location=[40.827332, -4.288363], zoom_start=12)
tile_provider = get_provider('ESRI_IMAGERY')
folium.TileLayer(tiles=tile_provider.url, attr='Esri, USGS').add_to(m)
# m = folium.Map(location=[40.827332,	-4.288363], tiles="Stamen Terrain", zoom_start=12)
crs = m.crs

# Imprimimos el sistema de referencia
print(crs)
# Obtenemos el sistema de referencia de los datos
crs = sim_geo.crs

# Imprimimos el sistema de referencia
print(crs)
# %% CLUSTER CON BOKEH
from bokeh.plotting import figure, show
from bokeh.tile_providers import get_provider
from bokeh.models import GeoJSONDataSource, ColumnDataSource#, Cluster
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap

# Crear una figura de Bokeh
from bokeh.plotting import figure, show

from pyproj import Proj, transform

inProj = Proj(init='epsg:4326') # sistema de coordenadas geográficas
outProj = Proj(init='epsg:3857') # sistema de coordenadas de Mercator

x, y = transform(inProj, outProj, -4.3, 40.8)
print(x, y)

n = 1000000
p = figure(title='Posiciones GPS',
            x_range=(-n+x, n+x),
            y_range=(-n+y, n+y),
            x_axis_type="datetime", 
            y_axis_type="datetime")
# p = figure()


# Añadimos el fondo
tile_provider = get_provider('ESRI_IMAGERY')
p.add_tile(tile_provider)

# Convertir el geodataframe en un conjunto de datos de Bokeh
geo_source = GeoJSONDataSource(geojson=gdf.to_json())

# Añadir los datos al gráfico como círculos
renderer = p.circle(x='x', y='y', source=geo_source, fill_color='color',
                    line_color='black', line_width=0.5, size=10)

p.add_layout(renderer)

# Mostrar la figura
show(p)



