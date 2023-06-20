# %% IMPORT LIBRERIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

import geolocalizacion.weather as weather
import geolocalizacion.geolocalizacion as geoloc
# %% IMPORT WEATHER DATA
weather_dict = weather.load_weather_dataframe()

# %% LOCATION_SELECTION
d = weather_dict['daily']
daily_condition = (d.location == 'Romangordo')
h = weather_dict['hourly']
hourly_condition = (h.location == 'Romangordo')
romangordo_dict = {'daily': d.loc[daily_condition], 
                   'hourly': h.loc[hourly_condition]}

# %% IMPORT BIRDS DATA

path = "E:\\duraton\\geolocalizacion\\_data"
filenames = geoloc.find_csv_filenames(path)[0:2]

info_archivos = list(map(geoloc.extract_info, filenames))
info_pajaros = pd.DataFrame(info_archivos, columns=['especie','ID','nombre'])
info_pajaros['color'] = pd.Series(['green', 'blue', 'purple', 'red', 'orange'])


df = geoloc.load_data(path, filenames, reindex_data=False)
df = df.merge(info_pajaros, how = 'left', on = 'ID')

# %% INDIVIDUAL SELECTION
condition = (df.nombre == 'Gato')
df_gato = df.loc[condition]
# condition = ((df2['UTC_datetime']>=pd.to_datetime('2021-01-01')) & (df2['UTC_datetime']<=pd.to_datetime('2021-03-31')))
# df2 = df2.loc[condition]