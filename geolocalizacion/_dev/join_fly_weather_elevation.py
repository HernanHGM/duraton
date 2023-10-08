# %% IMPORT LIBRARIES
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
from geolocalizacion.flying_discrimination import FlightAnalyzer


# %% IMPORT BIRDS DATA

path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw"
filenames = dp.find_csv_filenames(path)
nombre = 'Zorita'
filenames = [item for item in filenames if nombre in item]

df = dp.load_data(path, filenames, reindex_data=False)
# %% IMPORT WEATHER DATA
def get_closest_weather(df_fly, weather_dict):
    df_fly['closest_location'] = df_fly.apply(weather.find_nearest_location, 
                                              args=(weather_dict['coordinates'],), 
                                              axis=1)
    df_fly, _ = weather.join_fly_weather(weather_dict, df_fly, freq='hourly')
    df_fly, _ = weather.join_fly_weather(weather_dict, df_fly, freq='daily')
    return df_fly
weather_dict = weather.load_weather_dataframe()
df = get_closest_weather(df, weather_dict)
# %% CALCULATE FLYING POSITIONS
plt.close('all')
Fa = FlightAnalyzer(df)
x, freq, n_start, n_end = Fa.get_histogram(column='speed_km_h')
params, cov_matrix = Fa.optimize_parameters(x, freq, plot=True)
uncertain_values = Fa.find_flying_uncertainty(x, freq, params, plot=True)
df = Fa.define_flying_situation(uncertain_values)
df.boxplot('speed_km_h', by='flying_situation')
# %% LOAD & JOIN ELEVATION TERRAIN
el = ElevationLoader()
df_elevation = el.load_necessary_files(df)
fej = FlyElevationJoiner()

df_joined = fej.fly_elevation_join(df, df_elevation)
# %%
path_enriquecido = f"E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida\\{nombre}_enriquecida.csv"
df_joined.to_csv(path_enriquecido, index=False)
# %%
df_load = pd.read_csv(path_enriquecido)

# %% UNIMOS DATOS PUROS CON DATOS TIEMPO
freq = 'hourly'
df_fly_weather, weather_variables = geoloc.join_fly_weather(romangordo_dict, 
                                                            df_gato_fin, freq)

# fly_variables = ['Altitude_m', 'situacion', 'time_step_s', 'distance_2D']
# df_fly_weather = df_fly_weather[weather_variables + fly_variables]

