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
# %%
# path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw"
## IN case we want to use already enriched with elevation data
path = "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation"
filenames = dp.find_csv_filenames(path)
# To extract just the bird name
# names_list = [filename.replace('.csv', '') for filename in filenames]
names_list = [filename.replace('_elevation.csv', '') for filename in filenames]
# to remove certain individuals, or select them depends on the 'not'
names_list = [name for name in names_list if name in ('Navilla', 'Gato', 'Deleitosa')]



## %% IMPORT WEATHER DATA
weather_dict = weather.load_weather_dataframe()

# %% DEFINE PATHS
for name in names_list:  
    # path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw"
    # filenames = dp.find_csv_filenames(path)
    # filenames = [item for item in filenames if name in item]
 
    # ## %% IMPORT BIRDS DATA
    # df_fly = dp.load_preprocess_data(path, filenames, origin='movebank', reindex_data=False)
    
    # ## %% FILTER
    # '''
    # Due to massive amount of data with just 1s of timestamp, I am adding this filter
    # It is not supposed to be used in general, but in this special case is useful
    # to accelerate the prcess
    # '''
    # df_fly = df_fly[(df_fly.time_step_s>10)]
    # print('N registers: ', len(df_fly))
    
    # ## %% CALCULATE FLYING POSITIONS
    # plt.close('all')
    # Fa = FlightAnalyzer(df_fly)
    # x, freq, n_start, n_end = Fa.get_histogram(column='speed_km_h') 
    # params, cov_matrix = Fa.optimize_parameters(x, freq, plot=False)
    # uncertain_values = Fa.find_flying_uncertainty(x, freq, params, plot=False)
    # df_fly = Fa.define_flying_situation(uncertain_values)
    # df_fly.boxplot('speed_km_h', by='flying_situation')
    
    # ## %% PREDICT UNDEFINED FLYING VALUES
    # Ufc = UndefinedFlyClassifier()
    # df_fly = Ufc.train_model(df_fly)
    # df_fly.boxplot('speed_km_h', by='flying_situation')
    
    # ## %% LOAD & JOIN ELEVATION TERRAIN
    # el = ElevationLoader()
    # df_elevation = el.load_necessary_files(df_fly)
    # fej = FlyElevationJoiner()
    # df_fly_elevation= fej.fly_elevation_join(df_fly, df_elevation)
    
    # ## %% SAVE FLY & ELEVATION DATA
    # path_elevation = f"E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation\\{name}_elevation.csv"
    # df_fly_elevation.to_csv(path_elevation, index=False, encoding="ISO-8859-1")
    
    ## %% READ FLY & ELEVATION DATA
    path_elevation = f"E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation\\{name}_elevation.csv"
    df_fly_elevation = pd.read_csv(path_elevation,
                                    index_col=False, 
                                    encoding="ISO-8859-1")
    
    ## %% JOIN FLY & WEATHER DATA
    df_fly_elevation_weather = weather.get_closest_weather(df_fly_elevation, 
                                                            weather_dict)
    
    ## %% SAVE FLY & ELEVATION & WEATHER DATA
    path_elevation_weather = f"E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\{name}_elevation_weather.csv"
    df_fly_elevation_weather.to_csv(path_elevation_weather,
                                    index=False,
                                    encoding="ISO-8859-1")
    print(name, 'saved')



