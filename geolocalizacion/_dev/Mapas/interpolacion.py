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


df = g.load_data(path, filenames)

# condition = ((df['UTC_date']>=pd.to_datetime('2022-01-01')) & (df['UTC_date']<=pd.to_datetime('2022-12-31')))
# # condition = (gdf['UTC_time']<pd.to_datetime('08:00:00'))
# df = df[condition]

dg = df.groupby(['ID']).agg(start_date=('UTC_date', np.min), end_date=('UTC_date', np.max))
df = df.merge(dg, how = 'left', on = 'ID')
df = df.merge(info_pajaros, how = 'left', on = 'ID')
df.drop(labels = ['Unnamed: 23', 'Unnamed: 22'], axis = 1, inplace = True)
# %% SELECT
lista_nombres = ['Gato', 'Deleitosa']
df_gato_deli = g.get_same_data(df, lista_nombres)

lista_nombres = ['Gato', 'Deleitosa', 'Navilla']
df_gato_deli_navi = g.get_same_data(df, lista_nombres)

lista_nombres = ['Conquista', 'Zorita']
df_con_zor = g.get_same_data(df, lista_nombres)

# %% CALCULO KERNEL
# start = time.process_time_ns()/(10**9)
# # gdf = gdf_sec[gdf_sec.ID == 211981].copy()

# # Create thresholds
# kde_levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
# # kde_levels = [0.05, 0.5, 1]
# gdf_kernels = g.partial_kernels(kde_levels, df)
# gdf_kernels = gdf_kernels.merge(info_pajaros, on='ID', how = 'left')

# gdf_kernels = gdf_kernels.merge(dg, how = 'left', on = 'ID')
# gdf_kernels['start_date'] = gdf_kernels['start_date'].astype("string")#.dt.strftime('%Y-%m-%d %H:%M:%S')
# gdf_kernels['end_date'] = gdf_kernels['end_date'].astype("string")#.dt.strftime('%Y-%m-%d %H:%M:%S')
# color_especie={'Águila imperial': 'YlGn',
#                'Águila real': 'PuBu',
#                'Águila perdicera': 'OrRd'}
# esquema_color = [color_especie[especie] for especie in gdf_kernels.especie]
# gdf_kernels['color_map'] = esquema_color

# # c = g.total_kernels(kde_levels, gdf)

# # geo = geo.to_crs(epsg=3857)
# # # Calculate area
# # geo['area'] = geo['geometry'].area
# print('Tiempo transcurrido = ', time.process_time_ns() /
#       (10**9) - start, ' segundos')
# %% GUARDO KERNELS
# gdf_kernels.to_file("E:\\trabajo_pajaros\\geolocalización\\kernels\\partial_kernels.shp")
gdf_kernels = gpd.read_file("E:\\trabajo_pajaros\\geolocalización\\kernels\\partial_kernels.shp")
# %% REPRESENTACION KERNELS

# Creo mapa

start = time.process_time_ns()/(10**9)
base_map = folium.Map(location=[39.742117, -5.7012035], zoom_start=12)

for ID, subdf in gdf_kernels.groupby('especie'):
    g.add_kernels(subdf, base_map)
g.plot_map(base_map)
print('Tiempo transcurrido = ', time.process_time_ns() /
      (10**9) - start, ' segundos')

# %% MAPA ANIMADO

m = folium.Map(location=[39.29, -5.75], zoom_start=11)
g.add_animated_points(df_con_zor, m, df_kernels=gdf_kernels)

g.plot_map(m)

# %% PREPARO PARA CALCULO
columnas = ['UTC_datetime', 'Latitude', 'Longitude', 
            'Altitude_m', 'especie', 'nombre']
df_con = df_con_zor.loc[(df_con_zor.nombre == 'Conquista'), columnas]
df_con.set_index('UTC_datetime', inplace = True)
df_zor = df_con_zor.loc[(df_con_zor.nombre == 'Zorita'), columnas]
df_zor.set_index('UTC_datetime', inplace = True)


distancias = g.calculate_distance(df_con, df_zor)


# %%
plt.close('all')
distancias.plot(y = '3D')
distancias.plot(y = '2D')
distancias.plot(y = 'altura', kind = 'hist')

distancias.plot(x = 'altura', y = '2D', kind = 'kde')

# %% En proceso, Pvalor

df_con_zor.boxplot('Altitude_m', by=['nombre']) 
dfx = df_con_zor
import scipy.stats as stats
def get_ttest(x,y,sided=1):
    return stats.ttest_ind(x, y, equal_var=True).pvalue/sided
col_effect = 'Altitude_m'
v1 = df_zor.Altitude_m
v2 = df_con.Altitude_m
a = stats.ttest_ind(v1, v2, equal_var=False)
# a = get_ttest(
#     dfx.loc[dfx['nombre'] == 'Zorita', col_effect],
#     dfx.loc[dfx['nombre'] == 'Conquista', col_effect])

# %% Quitar alturas negativas
lista_nombres = ['Zorita']
df_zor = df_con_zor[df_con_zor.nombre== 'Zorita']
negative_values = df_zor.Altitude_m.le(0) 
altitude_median = df_zor.loc[~negative_values, 'Altitude_m'].median()
df_zor.loc[negative_values, 'Altitude_m'] = altitude_median
df_zor.boxplot('Altitude_m', by=['nombre']) 
# %% Calcular velocidades

# df_zor.plot(x = 'Altitude_m', y = 'speed_km_h', kind = 'scatter')
t = df_zor.UTC_datetime.diff().dt.seconds
t = t/3600
x = df_zor.Longitude.diff()
x2 = g.deg_km(x)
vx = x2/t
y = df_zor.Latitude.diff()
y2 = g.deg_km(y)
vy = y2/t
z = df_zor.Altitude_m.diff()
z2 = z/1000
vz = z2/t

v = np.sqrt(vx**2 + vy**2 + vz**2)

df_zor['V_x'] = vx
df_zor['V_y']= vy
df_zor['V_z'] = vz
df_zor['V'] = v
# %%

acc = np.sqrt(df_zor.acc_x**2 + df_zor.acc_y**2 + df_zor.acc_z**2)

mag = np.sqrt(df_zor.mag_x**2 + df_zor.mag_y**2 + df_zor.mag_z**2)


df_zor['mag'] = mag
df_zor['acc'] = acc
# %%
df_zor.boxplot('V_x', by=['nombre'])
df_zor.boxplot('V_y', by=['nombre'])
df_zor.boxplot('V_z', by=['nombre'])
df_zor.boxplot('V', by=['nombre'])

# %% kde plot
condition = ((df_zor['UTC_datetime']>=pd.to_datetime('2021-01-01')) & (df_zor['UTC_datetime']<=pd.to_datetime('2021-01-31')))
df_zor = df_zor[condition]


levels = [0.1, 0.3, 0.5, 0.7, 1]
kwargs = {'levels': levels,
  'fill': True,
  'cmap': 'Reds',
  'alpha': 0.9}
plt.close('all')
fig, (ax1, ax2) = plt.subplots(1, 2)
edit = sns.kdeplot(x=df_zor['speed_km_h'], y=df_zor['Altitude_m'], ax=ax1, **kwargs)
orig = sns.kdeplot(x=df_zor['V'], y=df_zor['Altitude_m'], ax=ax2, **kwargs)