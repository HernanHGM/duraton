# %% LIBRERIAS
import geopandas as gpd
import folium
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import webbrowser
from datetime import timedelta
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
info_pajaros['color'] = pd.Series(['green', 'blue', 'purple', 'red', 'orange'])


df = g.load_data(path, filenames, reindex_data = False)

# condition = ((df['UTC_date']>=pd.to_datetime('2022-01-01')) & (df['UTC_date']<=pd.to_datetime('2022-12-31')))
# # condition = (gdf['UTC_time']<pd.to_datetime('08:00:00'))
# df = df[condition]

dg = df.groupby(['ID']).agg(start_date=('UTC_date', np.min), end_date=('UTC_date', np.max))
df = df.merge(dg, how = 'left', on = 'ID')
df = df.merge(info_pajaros, how = 'left', on = 'ID')
df.drop(labels = ['Unnamed: 23', 'Unnamed: 22'], axis = 1, inplace = True)
# %% SELECT
df_zor = df.query('nombre == "Zorita"')
# %% CONVIERTO A GEOPANDAS
geometry = [Point(xy) for xy in zip(df_zor.Longitude, df_zor.Latitude)]
gdf = gpd.GeoDataFrame(df_zor, geometry=geometry, crs='epsg:4326')
# %%
import geopandas as gpd
from shapely.geometry import LineString, mapping
import fiona
import pyproj






condition = ((gdf['UTC_datetime']>pd.to_datetime('2020-07-16 12:00:00')) & 
             (gdf['UTC_datetime']<pd.to_datetime('2020-07-16 17:00:00')))

gdf2 = gdf.loc[condition, ['UTC_time', 'geometry']]

# unir todos los puntos en una sola línea
# line = gdf2.geometry.unary_union

# Crear una lista de coordenadas a partir de la columna 'geometry'
coords = list(gdf2['geometry'].apply(lambda x: x.coords[0]))

# Crear una línea a partir de la lista de coordenadas
line = LineString(coords)
# Crear un nuevo DataFrame con una sola fila que contiene la línea
line_zor = gpd.GeoDataFrame({'geometry': [line]})
#%%

import os, fiona

fiona.supported_drivers['KML'] = 'rw' #Enable kml driver
path = "E:\\trabajo_pajaros\\geolocalización"
out_folder = r"E:\trabajo_pajaros\geolocalización"

filename = os.path.join(out_folder, "zorita_points.kml")    
gdf2.to_file(filename, driver='KML')

filename = os.path.join(out_folder, "zorita_line.kml")    
line_zor.to_file(filename, driver='KML')

filename = os.path.join(out_folder, "zorita_line.json") 
line_zor.to_file(filename, driver='GeoJSON')

