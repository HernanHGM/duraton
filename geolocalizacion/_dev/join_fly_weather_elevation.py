# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

import geolocalizacion.weather as weather
import geolocalizacion.geolocalizacion as geoloc
import geolocalizacion.elevation as ele
# %% IMPORT WEATHER DATA
weather_dict = weather.load_weather_dataframe()

# %% IMPORT ELEVATION_DATA
file_path = 'E:\\duraton\\geolocalizacion\\_data\\elevation\\raw\\N39W006.hgt'
df_elevation = ele.load_hgt_file(file_path)
# %% LOCATION_SELECTION
d = weather_dict['daily']
daily_condition = (d.location == 'Romangordo')
h = weather_dict['hourly']
hourly_condition = (h.location == 'Romangordo')
romangordo_dict = {'daily': d.loc[daily_condition], 
                   'hourly': h.loc[hourly_condition]}

# %% IMPORT BIRDS DATA

path = "E:\\duraton\\geolocalizacion\\_data"
filenames = geoloc.find_csv_filenames(path)
filenames = [item for item in filenames if "Gato" in item]

info_archivos = list(map(geoloc.extract_info, filenames))
info_pajaros = pd.DataFrame(info_archivos, columns=['especie','ID','nombre'])
info_pajaros['color'] = pd.Series(['green', 'blue', 'purple', 'red', 'orange'])


df = geoloc.load_data(path, filenames, reindex_data=False, speed_limit=1)
df = df.merge(info_pajaros, how = 'left', on = 'ID')

# %% FILTER BIRD DATA
bird_condition = (df.nombre == 'Gato')
time_condition = (df.time_step_s < 60*15)
satelite_condition =  (df.satcount > 4)
flying_condition =  (df.situacion =='volando')
condition =  bird_condition & time_condition & satelite_condition# flying_condition
df_gato = df.loc[condition]
# condition = ((df2['UTC_datetime']>=pd.to_datetime('2021-01-01')) & (df2['UTC_datetime']<=pd.to_datetime('2021-03-31')))
# df2 = df2.loc[condition]  
# %% FILTER ELEVATION DATA
max_lat = max(df_gato.Latitude)
min_lat = min(df_gato.Latitude)
max_long = max(df_gato.Longitude)
min_long= min(df_gato.Longitude)

condition = (df_elevation.max_lat <= max_lat) &\
            (df_elevation.min_lat >= min_lat) &\
            (df_elevation.max_long <= max_long) &\
            (df_elevation.min_long >= min_long)
df_elevation = df_elevation.loc[condition]
# %% EXAMPLE CONDITIONAL_MERGE
import pandas as pd

# DataFrame con coordenadas geográficas concretas
df_coordenadas = pd.DataFrame({
    'latitud': [39.5684, 40.1234, 38.8765],
    'longitud': [-5.74272, -6.9876, -4.5678],
   })

# DataFrame con rangos
df_rangos = pd.DataFrame({
    'rango_lat_min': [39.0, 40.0, 38.5],
    'rango_lat_max': [39.9, 40.5, 39.0],
    'rango_lon_min': [-6.0, -7.0, -6.0],
    'rango_lon_max': [-5.5, -6.5, -4.0],
    'valor_rango': [10, 20, 30]  # Datos adicionales que quieres conservar del segundo DataFrame
})



# Consulta SQL para realizar la unión
query = """
SELECT *
FROM df_coordenadas AS c
LEFT JOIN df_rangos AS r
ON c.latitud BETWEEN r.rango_lat_min AND r.rango_lat_max
AND c.longitud BETWEEN r.rango_lon_min AND r.rango_lon_max
"""
import pandas as pd
from pandasql import sqldf

# Ejecutar la consulta usando pandasql
df_resultado = sqldf(query, locals())

print(df_resultado)
# %% ELEVATION BIRD CONDITIONAL MERGE
import pandas as pd
from pandasql import sqldf
# Consulta SQL para realizar la unión
query = """
SELECT *
FROM df_gato AS df1
LEFT JOIN df_elevation AS df2
ON df1.Latitude BETWEEN df2.min_lat AND df2.max_lat
AND df1.Longitude BETWEEN df2.min_long AND df2.max_long
"""


# Ejecutar la consulta usando pandasql
df_resultado = sqldf(query, locals())

df_gato_fin = df_resultado.drop(labels=['ID', 'datatype', 'hdop',
                                        'Unnamed: 22',  'min_long', 'max_long',
                                        'min_lat', 'max_lat'], axis=1)
# %% ANÁLISIS CALIDAD DATOS
df_gato_fin = df_gato_fin.assign(bird_altitude=df_gato_fin.Altitude_m - df_gato_fin.elevation) 
df_gato_fin = df_gato_fin.loc[df_gato_fin['bird_altitude']>0]
# df_gato.plot(x='Altitude_m', y='speed_km_h')
# %% UNIMOS DATOS PUROS CON DATOS TIEMPO
freq = 'hourly'
df_fly_weather, weather_variables = geoloc.join_fly_weather(romangordo_dict, 
                                                            df_gato_fin, freq)

# fly_variables = ['Altitude_m', 'situacion', 'time_step_s', 'distance_2D']
# df_fly_weather = df_fly_weather[weather_variables + fly_variables]
# %% GUARDO O CARGO DATOS DE VUELO, ELEVACION Y METEOROLÓGICO
path_enriquecido = "E:\\duraton\\geolocalizacion\\_data\\Gato_enriquecida.csv"
# df_fly_weather.to_csv(path_enriquecido)
df_fly_weather = pd.read_csv(path_enriquecido)
df_fly_weather['acc'] = np.sqrt(df_fly_weather.acc_x**2 +
                                df_fly_weather.acc_y**2 +
                                df_fly_weather.acc_z**2)

df_fly_weather['mag'] = np.sqrt(df_fly_weather.mag_x**2 +
                                df_fly_weather.mag_y**2 +
                                df_fly_weather.mag_z**2)

