# %% LIBRERIAS
import geopandas as gpd
import folium
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import webbrowser
from shapely.geometry import MultiPolygon, Polygon, Point
from folium.plugins import MarkerCluster, HeatMap
from folium import plugins
from funciones_pajaros import geolocalizacion as g

import time
start = time.process_time_ns()/(10**9)
print('Tiempo transcurrido = ', time.process_time_ns()/(10**9) - start, ' segundos')
# %% CARGO DATOS
file = 'E:\\trabajo_pajaros\\geolocalización\\data1.csv'
df = pd.read_csv(file)
df.UTC_time = pd.to_datetime(df.UTC_time)
# condition1 = (df['UTC_time']<pd.to_datetime('14:00:00')) & (df['UTC_time']>pd.to_datetime('12:00:00'))
# condition2 = (df['UTC_time']<pd.to_datetime('09:00:00'))
# df1 = df[condition1]
# df2 = df[condition2]



# df1['ID'] = 1
df1 = df.copy()
df1['ID'] = 1
df2 = df1.copy()
df2['Latitude'] = df2['Latitude']+0.08
df2['ID'] = 2
df = pd.concat([df1, df2])
# df1 = df[df[condition1]]
# %% CONVIERTO A GEOPANDAS
geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs = 'epsg:4326')
print(gdf.crs)
n_id = gdf['ID'].nunique()
ids = gdf['ID'].unique()

# %% CREO ZONAS DE INFLUENCIA ENTORNO A PUTNOS
points = [(-4.288363, 40.827332), (-4.288363, 40.8225)]
# points = df.loc[:2]
antenas = g.circular_area(points, 2)
# %% PLOT WITH FOLIUM
# Creo mapa
base_map = folium.Map(location=[40.827332,	-4.288363], zoom_start=12)

col_name= 'ID'
# g.add_cluster(gdf, col_name, base_map)
# g.add_heatmap(gdf, col_name, base_map)
# # Necesita un dataframe que tenga la columna 'geometry' y la columna 'nombre'
# g.add_geometry(antenas, base_map, 'orange')
g.plot_map(base_map)
# def plot_map(m):
#     file_map = "E:\\trabajo_pajaros\\geolocalización\\map.html"
#     m.save(file_map)
#     webbrowser.open(file_map)

# %% INTERSECTION
# # Verifica si los puntos de gdf intersectan con los polígonos de polygon
# intersection = gdf.geometry.intersects(polygon.geometry[0])

# # Cuenta el número de puntos que intersectan con el polígono
# num_points_in_polygon = intersection.sum()

# Crea una lista para almacenar el número de puntos en cada polígono
points_in_polygons = []

# Itera sobre cada polígono en polygon
for i in range(len(antenas)):
  # Verifica si los puntos de gdf intersectan con el polígono actual
  intersection = gdf.geometry.intersects(antenas.geometry[i])
  
  # Cuenta el número de puntos que intersectan con el polígono
  num_points_in_polygon = intersection.sum()
  
  # Agrega el resultado a la lista
  points_in_polygons.append(num_points_in_polygon)

# Muestra la lista con el número de puntos en cada polígono
print(points_in_polygons)
# %% CALCULO KERNEL


# Create thresholds
kde_levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]

def partial_kernels(kde_levels, gdf):
    if 1 not in kde_levels:
        kde_levels.append(1)
    sns_kernels = g.get_sns_kernels(kde_levels, gdf) 
    gdf_kernels = g.get_gdf_kernels(sns_kernels, kde_levels)
    gdf_kernels['name'] = gdf_kernels['ID'].astype(str) +"_"+ gdf_kernels["level"].astype(str)
    
    return gdf_kernels
gdf_kernels = partial_kernels(kde_levels, gdf)
def total_kernels(kde_levels, gdf):
    if 1 in kde_levels:
        kde_levels = kde_levels[:-1]
    kernels_list = [[level, 1] for level in kde_levels]

    a = []
    for kde in kernels_list:
        a.append(partial_kernels(kde, gdf))
    b = pd.concat(a)
    return b
c = total_kernels(kde_levels, gdf)
        
# geo = geo.to_crs(epsg=3857) 
# # Calculate area
# geo['area'] = geo['geometry'].area
# %% REPRESENTACION KERNELS

# Creo mapa

base_map = folium.Map(location=[40.827332,	-4.288363], zoom_start=12)
plot_gdf = c[c.level==0.5] #[gdf_kernels['ID']==2]

g.add_cluster(gdf, col_name, base_map)
# g.add_geometry(plot_gdf, base_map, 'orange')
g.add_kernels(gdf_kernels, base_map)
g.plot_map(base_map)

