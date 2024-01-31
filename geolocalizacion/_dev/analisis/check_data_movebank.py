import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

import geolocalizacion.weather as weather
import geolocalizacion.data_processing as dp
from geolocalizacion.elevation import ElevationLoader, FlyElevationJoiner
from geolocalizacion.flying_discrimination import FlightAnalyzer, UndefinedFlyClassifier

# %% Cargo todos los csv a comparar

# Carpeta con los csvs preliminares con los que se trabajó
path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw"
filenames = dp.find_csv_filenames(path)
df_ornitela = dp.load_data(path, filenames, reindex_data=False)

# Carpeta con los csvs completos con los que se trabajo
path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw\\completos"
filenames = dp.find_csv_filenames(path)
df_ornitela2 = dp.load_data(path, filenames, reindex_data=False)

# Carpeta con los csvs completos con los que se trabajo, una vez añadida la elevación
path = "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation"
filenames = dp.find_csv_filenames(path)
df_ornitela3 = dp.load_data(path, filenames, pre_process=False, reindex_data=False)

# Archivo csv con los datos descargados de movebank
full_filenames = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw\\movebank\\Campo Arauelo_Rapaces.csv" 
df_movebank = pd.read_csv(full_filenames,  
                  index_col=False,
                  encoding="utf-8")
# %% Muestro las fechas de inicio y fin para cada dataframe a ver si casan

names_list = list(df_ornitela.name.unique())
for name in names_list:
    df_1 = df_ornitela[df_ornitela['name']==name]
    df_2 = df_movebank[df_movebank['individual-local-identifier']==name]
    df_3 = df_ornitela2[df_ornitela2['name']==name]
    
    df_4 = df_ornitela3[df_ornitela3['name']==name]


    df3 = df_2.copy()
    df3['timestamp'] = pd.to_datetime(df3['timestamp'], format='%Y%m%d %H:%M:%S')
    print('Nuevo: ', name, min(df3['timestamp']), max(df3['timestamp']))
    print('Antiguo: ', name, min(df_1['UTC_datetime']), max(df_1['UTC_datetime']))
    print('Antiguo completo: ', name, min(df_3['UTC_datetime']), max(df_3['UTC_datetime']))
    print('elevation: ', name, min(df_4['UTC_datetime']), max(df_4['UTC_datetime']))
    print()
# %%
df4 = pd.merge(df_1, df3, how='inner', left_on='UTC_datetime', right_on='timestamp')

# %%
df4.plot('speed_km_h', 'ground-speed', title=name)
