# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:04:08 2023

@author: HernanHGM

1.- AGRUPADO FIJO
Creo Agrupado de 1 hora de hora en hora para tener datos agrupados
con un intervalo lo suficientemente grande como para ser fiable y lo 
suficientemente pequeño como para que la media no suavice en exceso los datos.

En verdad es una hora porque ya tenía una columna con la hora modo hh:00:00
Lo suyo habría sido explorar distintos intervalos, pero los resultados dudo 
que mejoren mucho y supone una complicación relevante

2.- AGRUPADO MOVIL
Además se ha estudiado el hacer la media móvil dato a dato, es decir para cada
dato hacer la media con las siguiente X horas de datos, pero eso implica dos 
problemas: datos duplicados, e intervalos con diferente numero de datos.
    - Datos duplicados porque el datos de las 15:10:00 aparecera para 15:00:00, 
      15:05:00, y 15:10:00 mienttras que 15:50:00 aparecerá 11 veces 
    - Además los datos de últimas horas del día tendrán menos y menos datos 
      porque a partir de las 2:00:00 genero un intervalo de 3 horas por ejemplo
      pero no hay más que el dato de las 2:00:00 y el de las 2:05:00 y se 
      cortan los datos hasta el día siguiente
"""
# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
# %% LOAD PREPARED FILE
path_load = "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\all_data.csv"
# path_load = "E:/duraton/geolocalizacion/_data/fly/interpolated/all_interpolated_data_distances.csv"
date_columns = ['UTC_datetime']
df = pd.read_csv(path_load,
                 parse_dates = date_columns,
                 index_col=False, 
                 encoding="ISO-8859-1")
# %% check if distances already appears for the interpolated situation
distances = ['km_Conquista', 'km_Deleitosa', 'km_Gato', 'km_Navilla', 'km_Zorita']
if all(x in df.columns for x in distances):
    print('appears distances')
else:
    distances = []
# %% DEFINE COLUMNS
gby_columns = ['specie', 'name', 'month_name', 
               'month_number', 'week_number', 
               'UTC_date', 'hour',
               'flying_situation']
mean_columns = ['Latitude', 'Longitude', 'Altitude_m', 
                'mag_x', 'mag_y', 'mag_z', 'mag',
                'acc_x', 'acc_y', 'acc_z', 'acc', 
                'bird_altitude', 'speed_km_h',
                'elevation', 'direction_deg',
                'tempC', 'DewPointC',
       'windspeedKmph', 'pressure', 'visibility', 'cloudcover', 'precipMM',
       'humidity', 'maxtempC', 'mintempC', 'avgtempC', 'sunHour',
       'totalSunHour', 'uvIndex'] + distances

sum_columns = ['time_step_s', 'distance_2D']


# %% CREATE DICTIONARIES
mean_dict = {col: 'mean' for col in mean_columns}
sum_dict = {col: 'sum' for col in sum_columns}

# To measure the differences in height we create two metrics, the sum of the 
# positive increments, and the commo sum were all values are considered
def positive_sum(x):
    suma = x[x>0].sum()
    return suma

altitude_dict = {'distance_height': [positive_sum, np.sum]}

agg_dict = {}
agg_dict.update(mean_dict)
agg_dict.update(sum_dict)    
agg_dict.update(altitude_dict)    
# %% FIXED GROUPBY
dg = df.groupby(gby_columns)\
    .agg(agg_dict)\
    .rename(columns={'time_step_s': 'flying_time',
                     'distance_2D': 'distance_travelled',
                     'distance_height': 'distance_ascended'})\
    .reset_index()

dg.columns = dg.columns.map('_'.join)\
               .str.strip('_')\
               .str.replace('_mean', '')\
               .str.replace('_sum', '') 

# %% CALCULATE TOTAL TIME
join_columns = [col for col in gby_columns if col != 'flying_situation']

dg2 = dg.groupby(join_columns)\
    .agg({'flying_time': 'sum'})\
    .rename(columns={'flying_time': 'total_time'})\
    .reset_index() 
# %% TOTAL METRICS
df_join = pd.merge(dg, dg2, how='left', on=join_columns)

df_join['flying_time_percentage'] = 100*df_join['flying_time']/df_join['total_time']
df_join['distance_flied_by_hour'] = 3600*df_join['distance_travelled']/df_join['flying_time']
df_join['distance_ascended_by_hour'] = 3600*df_join['distance_ascended']/df_join['flying_time']
df_join['distance_ascended_positive_by_hour'] = 3600*df_join['distance_ascended_positive']/df_join['flying_time']

# %% ANALYSIS DATA
# There should be just one register per aggrupation because it is the Primay Key
analysis = df_join.groupby(['name', 
                            'UTC_date',
                            'hour',
                            'flying_situation']).size()
# %% SAVE DATA                           
save_path = "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\all_grouped_data.csv"
# save_path = "E:\\duraton\\geolocalizacion\\_data\\fly\\interpolated\\all_interpolated_grouped_data.csv"
df_join.to_csv(save_path,
               index=False,
               encoding="ISO-8859-1")
# %% MOVING GROUPBY (UNUSED)

ma_2h = df\
    .set_index('UTC_datetime')\
    .sort_index()\
    .groupby('flying_situation')\
    .rolling('6H')\
    .agg(agg_dict)\
    .rename(columns={'time_step_s': 'flying_time',
                     'distance_2D': 'distance_travelled'})\
    .reset_index()
    
# %% (UNUSED)

a = df_join.loc[(df_join['UTC_date']=='2020-11-29')&
                (df_join['hour']=='11:00:00')&
                (df_join['name']=='Conquista')]