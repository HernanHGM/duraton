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


df = g.load_data(path, filenames, reindex_data=False)
df = df.merge(info_pajaros, how = 'left', on = 'ID')

# %% SELECCION INDIVIDUO
condition = (df.nombre == 'Zorita')
df2 = df.loc[condition]
condition = ((df2['UTC_datetime']>=pd.to_datetime('2021-01-01')) & (df2['UTC_datetime']<=pd.to_datetime('2021-03-31')))
df2 = df2.loc[condition]
# %% PINTO DATOS

levels = [0.1, 0.3, 0.5, 0.7, 1]
kwargs = {'levels': levels,
  'fill': True,
  'cmap': 'Reds',
  'alpha': 0.9}
plt.close('all')
# fig, (ax1) = plt.subplots(1, 1)
# edit = sns.kdeplot(x=df2['speed_km_h'], ax=ax1, **kwargs)

df2['speed_km_h'].plot(kind = 'kde')

# %% MODELO CLASIFICACION 1
from sklearn.mixture import GaussianMixture
X = np.array(df2['speed_km_h']).reshape(-1, 1)
gm = GaussianMixture(n_components=2, random_state=0).fit(X)
print(gm.means_)


print(gm.predict([[0], [1.53], [1.54], [3]]))
# %%
condition = (df.nombre == 'Zorita')
df_zor = df.loc[condition]
X = np.array(df_zor['speed_km_h']).reshape(-1, 1)

# clasificar las filas del DataFrame
labels = gm.predict(X)
df_zor['clasificacion'] = np.where(labels==0, 'posado', 'volando')
# %%
import matplotlib.pyplot as plt

# Obtener las probabilidades de pertenencia a cada clase para cada valor de 'speed_km_h'
x = np.linspace(0,3,1000).reshape(-1, 1)
probs = gm.predict_proba(x)*0.5
limite_index = np.abs(probs[:,0] - probs[:,1]).argmin()

# Obtener el valor límite correspondiente a ese índice
limite = x[limite_index]


# Graficar las probabilidades de pertenencia a cada clase en función de 'speed_km_h'
df2['speed_km_h'].plot(kind = 'hist', bins = 100, density = True)
plt.axvline(x=limite, color='black', label='Valor límite')
plt.legend()
plt.xlabel('speed_km_h')
plt.ylabel('Probabilidad de pertenencia a cada clase')
plt.show()

# %% MODELO CLASIFICACION 2
from sklearn.mixture import BayesianGaussianMixture
data = np.array(df2['speed_km_h'])
data_neg = -data

# Unir los datos originales y sus negativos
data_all = np.concatenate((data, data_neg)).reshape(-1, 1)
gm = GaussianMixture(n_components=3, random_state=0).fit(data_all)
print(gm.means_)
df_all = pd.DataFrame(data_all)
df_all.plot(kind = 'kde')
# print(gm.predict([[0], [1.53], [1.54], [3]]))
# %%
condition = (df.nombre == 'Zorita')
df_zor = df.loc[condition]
X = np.array(df_zor['speed_km_h']).reshape(-1, 1)

# clasificar las filas del DataFrame
labels = gm.predict(X)
df_zor['clasificacion'] = np.where(labels==0, 'posado', 'volando')
# %%
import matplotlib.pyplot as plt

# Obtener las probabilidades de pertenencia a cada clase para cada valor de 'speed_km_h'
x = np.linspace(-10,10,1000).reshape(-1, 1)
probs = gm.predict_proba(x)
limite_index = np.abs(probs[:,0] - probs[:,1]).argmin()
plt.plot(x, probs[:,0], 'b')
plt.plot(x, probs[:,1], 'r')
plt.plot(x, probs[:,2], 'k')
#%%

# Obtener el valor límite correspondiente a ese índice
limite = x[limite_index]


# Graficar las probabilidades de pertenencia a cada clase en función de 'speed_km_h'
df2['speed_km_h'].plot(kind = 'hist', bins = 100, density = True)
plt.axvline(x=limite, color='black', label='Valor límite')
plt.plot(x, probs[:,0], 'b')
plt.plot(x, probs[:,1], 'r')
plt.plot(x, probs[:,2], 'k')
plt.legend()
plt.xlabel('speed_km_h')
plt.ylabel('Probabilidad de pertenencia a cada clase')
plt.show()
