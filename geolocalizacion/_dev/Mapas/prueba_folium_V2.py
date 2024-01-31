# %% LIBRERIAS
import geopandas as gpd
import folium
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser
from shapely.geometry import Polygon, Point
from folium.plugins import MarkerCluster
import time
# %% CARGO DATOS
file = 'E:\\trabajo_pajaros\\geolocalización\\data1.csv'
df = pd.read_csv(file)

# %% CONVIERTO A GEOPANDAS
geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)
# %% MEJORA, USO FUNCION DE GPD PARA CREAR CIRCULOS

points = [(-4.288363, 40.827332), (-4.288363, 40.8225)]
points2 = df.loc[:2]

def circular_area(data, r_plane):
    import geopandas as gpd
    import pandas as pd
    from math import pi
    from shapely.geometry import Point
    
    if isinstance(data, list) == True:
        geometry = [Point(xy) for xy in data]  
    elif isinstance(data, pd.DataFrame) == True:
        geometry = [Point(xy) for xy in zip(data.Longitude, data.Latitude)]
    
    # Convertimos datos a formato geopandas
    points = gpd.GeoDataFrame(data, geometry=geometry)
    
    # Crea un buffer de r_plane km alrededor de cada punto
    r_plane = 0.1
    r_deg = 360*r_plane/(6371*2*pi)
    points_buffer = points.geometry.buffer(r_deg)

    # Crea un GeoDataFrame con los buffers
    buffered_points = gpd.GeoDataFrame(geometry=points_buffer)
    
    n = len(buffered_points)
    names = list(map(lambda x: 'antena ' + str(x+1), range(n)))
    buffered_points['nombre'] = names   
    return buffered_points

buffered_points = circular_area(points2, 0.1)
# %%
m = folium.Map(location=[40.827332,	-4.288363], zoom_start=12)

# create a marker cluster called "Public toilet cluster"
marker_cluster = MarkerCluster().add_to(m)

for _, r in buffered_points.iterrows():
    # Without simplifying the representation of each borough,
    # the map might not be displayed
    sim_geo = gpd.GeoSeries(r['geometry'])
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j,
                           style_function=lambda x: {'fillColor': 'orange'})
    folium.Popup(r['nombre']).add_to(geo_j)
    geo_j.add_to(m)
    
for _, r in gdf.iterrows():
    # Without simplifying the representation of each borough,
    # the map might not be displayed
    sim_geo = gpd.GeoSeries(r['geometry'])
    geo_j = folium.GeoJson(data=sim_geo.to_json())
    geo_j.add_to(marker_cluster)


    
file_map = "E:\\trabajo_pajaros\\geolocalización\\map.html"
m.save(file_map)

webbrowser.open(file_map)

# %%
# # Verifica si los puntos de gdf intersectan con los polígonos de polygon
# intersection = gdf.geometry.intersects(polygon.geometry[0])

# # Cuenta el número de puntos que intersectan con el polígono
# num_points_in_polygon = intersection.sum()

import geopandas as gpd

# Crea una lista para almacenar el número de puntos en cada polígono
points_in_polygons = []

# Itera sobre cada polígono en polygon
for i in range(len(points_buffer)):
  # Verifica si los puntos de gdf intersectan con el polígono actual
  intersection = gdf.geometry.intersects(points_buffer.geometry[i])
  
  # Cuenta el número de puntos que intersectan con el polígono
  num_points_in_polygon = intersection.sum()
  
  # Agrega el resultado a la lista
  points_in_polygons.append(num_points_in_polygon)

# Muestra la lista con el número de puntos en cada polígono
print(points_in_polygons)
