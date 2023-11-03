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
from geolocalizacion.flying_discrimination import FlightAnalyzer, UndefinedFlyClassifier

# %% DEFINE PATHS
path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw"
filenames = dp.find_csv_filenames(path)
nombre = 'Navilla'
filenames = [item for item in filenames if nombre in item]

# %% IMPORT BIRDS DATA
df_fly = dp.load_data(path, filenames, reindex_data=False)
'''
IDs
conquista = 192663
Deleitosa = 201255
Gato = 213861
Navilla = 211981
zorita = 201254
'''

# %% CALCULATE FLYING POSITIONS
plt.close('all')
Fa = FlightAnalyzer(df_fly)
x, freq, n_start, n_end = Fa.get_histogram(column='speed_km_h')
params, cov_matrix = Fa.optimize_parameters(x, freq, plot=True)
uncertain_values = Fa.find_flying_uncertainty(x, freq, params, plot=True)
df_fly = Fa.define_flying_situation(uncertain_values)
df_fly.boxplot('speed_km_h', by='flying_situation')
# %% PREDICT UNDEFINED FLYING VALUES
Ufc = UndefinedFlyClassifier()
df_fly = Ufc.train_model(df_fly)
df_fly.boxplot('speed_km_h', by='flying_situation')
# %% LOAD & JOIN ELEVATION TERRAIN
el = ElevationLoader()
df_elevation = el.load_necessary_files(df_fly)
fej = FlyElevationJoiner()

df_fly_elevation= fej.fly_elevation_join(df_fly, df_elevation)

# %% SAVE FLY & ELEVATION DATA
path_elevation = f"E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation\\{nombre}_elevation.csv"

df_fly_elevation.to_csv(path_elevation, index=False, encoding="ISO-8859-1")
# %% READ FLY & ELEVATION DATA
df_fly_elevation = pd.read_csv(path_elevation,
                               index_col=False, 
                               encoding="ISO-8859-1")

# %% IMPORT WEATHER DATA
weather_dict = weather.load_weather_dataframe()

# %% JOIN FLY & WEATHER DATA
df_fly_elevation_weather = weather.get_closest_weather(df_fly_elevation, 
                                                       weather_dict)

# %% SAVE FLY & ELEVATION & WEATHER DATA
path_elevation_weather = f"E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\{nombre}_elevation_weather.csv"
df_fly_elevation_weather.to_csv(path_elevation_weather,
                                index=False,
                                encoding="ISO-8859-1")
