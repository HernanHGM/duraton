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
# %% def circular_coordinates()

def circular_coordinates(longitude, latitude, r_plane, n):
    import math
    pi = math.pi
    import numpy as np
    r_deg = 360*r_plane/(6371*2*math.pi)
    x_center = longitude
    y_center = latitude
    coordinates_list = np.zeros((n,2))
    angle = np.array([2 * pi * i/n for i in range (1, n+1)])
    coordinates_list[:,0] = x_center + r_deg * np.cos(angle)
    coordinates_list[:,1] = y_center + r_deg * np.sin(angle)
    return coordinates_list


# %% POLIGONO DE LAS COORDENADAS DE LA ANTENA
longitude, latitude = -4.288363, 40.827332
coordinates = circular_coordinates(longitude, latitude, 0.1, 50)
polygon_geom = Polygon(coordinates)
polygon1 = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])  

longitude, latitude = -4.288363, 40.8225
coordinates = circular_coordinates(longitude, latitude, 0.1, 50)
polygon_geom = Polygon(coordinates)
polygon2 = gpd.GeoDataFrame(index=[1], crs='epsg:4326', geometry=[polygon_geom])  

polygon = pd.concat([polygon1,polygon2])
# polygon['name'] = ['antena1']
# %% MEJORA, USO FUNCION DE GPD PARA CREAR CIRCULOS
import geopandas as gpd

# a = df.loc[:2]
# geometry = [Point(xy) for xy in zip(a.Longitude, a.Latitude)]
a = [(-4.288363, 40.827332), (-4.288363, 40.8225)]
geometry = [Point(xy) for xy in a]
points = gpd.GeoDataFrame(a, geometry=geometry)
# Crea un buffer de 1 km alrededor de cada punto
import math
r_plane = 0.1
r_deg = 360*r_plane/(6371*2*math.pi)
points_buffer = points.geometry.buffer(r_deg)

# Crea un GeoDataFrame con los buffers
buffered_points = gpd.GeoDataFrame(geometry=points_buffer)

# # Gráfica los puntos y los buffers
# buffered_points.plot()
# points.plot()

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
    # folium.Popup(r['name']).add_to(geo_j)
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
