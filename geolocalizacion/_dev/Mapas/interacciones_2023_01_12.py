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
from funciones_pajaros import f as f
import os
import time

# %% CARGO DATOS

path = "E:\\trabajo_pajaros\\geolocalización"
filenames = g.find_csv_filenames(path)

info_archivos = list(map(g.extract_info, filenames))
info_pajaros = pd.DataFrame(info_archivos, columns=['especie','ID','nombre'])

df = g.load_data(path, filenames)
df.drop(labels = ['Unnamed: 23', 'Unnamed: 22'], axis = 1, inplace = True)
# condition = ((df['UTC_date']>=pd.to_datetime('2022-01-01')) & (df['UTC_date']<=pd.to_datetime('2022-12-31')))
# # condition = (gdf['UTC_time']<pd.to_datetime('08:00:00'))
# df = df[condition]
df = df.merge(info_pajaros, how = 'left', on = 'ID')
dg = df.groupby(['ID']).agg(start_date=('UTC_date', np.min), end_date=('UTC_date', np.max))
df = df.merge(dg, how = 'left', on = 'ID')


# %% CONVIERTO A GEOPANDAS
geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:4326')

# %% CREO ZONAS DE INFLUENCIA ENTORNO A PUNTOS
points = [(-4.288363, 40.827332), (-4.288363, 40.8225)]
# points = df.loc[:2]
antenas = g.circular_area(points, 2)
# %% PLOT WITH FOLIUM
# Creo mapa
base_map = folium.Map(location=[40.827332,	-4.288363], zoom_start=12)

col_name = 'ID'
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
start = time.process_time_ns()/(10**9)
# gdf = gdf_sec[gdf_sec.ID == 211981].copy()

# Create thresholds
kde_levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
# kde_levels = [0.05, 0.5, 1]
gdf_kernels = g.partial_kernels(kde_levels, gdf)
gdf_kernels = gdf_kernels.merge(info_pajaros, on='ID', how = 'left')

gdf_kernels = gdf_kernels.merge(dg, how = 'left', on = 'ID')
gdf_kernels['Start_date'] = gdf_kernels['start_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
gdf_kernels['End_date'] = gdf_kernels['end_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
# c = g.total_kernels(kde_levels, gdf)

# geo = geo.to_crs(epsg=3857)
# # Calculate area
# geo['area'] = geo['geometry'].area
print('Tiempo transcurrido = ', time.process_time_ns() /
      (10**9) - start, ' segundos')
# %% REPRESENTACION KERNELS

# Creo mapa

start = time.process_time_ns()/(10**9)
base_map = folium.Map(location=[39.742117, -5.7012035], zoom_start=12)
# plot_gdf = c[c.level==0.5] #[gdf_kernels['ID']==2]
# col_name = 'ID'
# g.add_cluster(gdf, col_name, base_map)
# g.add_geometry(plot_gdf, base_map, 'orange')

g.add_kernels(gdf_kernels, base_map)
g.plot_map(base_map)
print('Tiempo transcurrido = ', time.process_time_ns() /
      (10**9) - start, ' segundos')
# %% SAVE KERNELS
gdf_kernels.to_file("E:\\trabajo_pajaros\\geolocalización\\kernels.shp")  
# %% PLOT POINTS VS TIME
import folium
from folium.plugins import TimeSliderChoropleth
start = time.process_time_ns()/(10**9)
base_map = folium.Map(location=[39.742117, -5.7012035], zoom_start=12)

def bird_map(gdf, m):

    # create a feature group for the bird data
    birds = folium.FeatureGroup(name='Birds')

    # create a color dictionary for the birds
    colors = {211981: 'red', 192663: 'blue', 213861: 'green', 201254: 'purple', 201255: 'orange'}

    # iterate through the bird data and add markers to the map
    for _, row in gdf.iterrows():
        color = colors[row['ID']]
        folium.Marker(location=[row['Latitude'], row['Longitude']], 
                      icon=folium.Icon(color=color),
                      popup=f'Bird ID: {row["ID"]} <br> Time: {row["UTC_datetime"]}').add_to(birds)

    # add the bird feature group to the map
    birds.add_to(m)
    # add time slider to map
    gdf['UTC_datetime'] = gdf['UTC_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    styledict = {211981: {'color': 'red'},
             192663: {'color': 'blue'},
             213861: {'color': 'green'},
             201254: {'color': 'purple'},
             201255: {'color': 'orange'},
            }

    TimeSliderChoropleth(data=gdf.to_json(),
                         styledict=styledict,
                         overlay=True, 
                         name='Timeline').add_to(m)

    # add the legend to the map
    folium.map.LayerControl('topright', collapsed=False).add_to(m)

condition = ((df['UTC_time']>pd.to_datetime('12:50:00')) & (df['UTC_time']<pd.to_datetime('13:00:00')))
# condition = (gdf['UTC_time']<pd.to_datetime('08:00:00'))
gdf2 = gdf[condition]
gdf2 = gdf2[['UTC_datetime', 'Latitude', 'Longitude', 'ID']]

bird_map(gdf2, base_map)
print('Tiempo transcurrido = ', time.process_time_ns() /
      (10**9) - start, ' segundos')
g.plot_map(base_map)

# %% PLOT POINTS

start = time.process_time_ns()/(10**9)
base_map = folium.Map(location=[39.742117, -5.7012035], zoom_start=10)

g.add_markers(gdf, base_map)

g.plot_map(base_map)
print('Tiempo transcurrido = ', time.process_time_ns() /
      (10**9) - start, ' segundos')
# %% SAVE POINTS
gdf['UTC_datetime'] = gdf['UTC_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
gdf['UTC_date'] = gdf['UTC_date'].dt.strftime('%Y-%m-%d')
gdf['UTC_time'] = gdf['UTC_time'].dt.strftime('%H:%M:%S')
gdf = gdf[['ID', 'UTC_datetime', 'UTC_date', 'UTC_time', 'Latitude', 'Longitude',
           'Altitude_m', 'temperature_C', 'speed_km_h', 'especie', 'nombre',
           'geometry']]
gdf.to_file("E:\\trabajo_pajaros\\geolocalización\\points.shp")  
