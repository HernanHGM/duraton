# %% FILE DESCRIPTION
"""
Created on Wed Oct 23 9:20:00 2023

@author: HernanHGM

1.-INDIVIDUAL BIRDS
Load the enriched data from all the birds and interpolate them to 5 minutes
Due to interpolation all str columns are set to none
drop 'specie', 'name', 'color', 'flying_situation', 'closest_location'
columns and apply enrichement method to recover str columns
'closest_location' is created as null because the weather info comes from 
interpolation not real village weather data. 
(And takes a lot to add the closest village name)
Apply fly classifier to non interpolated df to ensure flying and landed speed 
limits are the same for interpolated and non interpolated df
Define flying situation for interpolated df creating a ML model with the 
interpolated values, here we cannot apply noninterpolated_df
Save the interpolated data in one file for each bird

2.- ALL BIRDS
Load all interpolated birds data, filter it and save it in one unified file

3.- INTERACTING BIRDS
Load the unified file and save only the data from the interacting Birds
"""
# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

path_duraton = 'E:\\duraton'
if path_duraton not in sys.path:
    sys.path.append(path_duraton)
    
import geolocalizacion.data_processing as dp
from geolocalizacion.flying_discrimination import FlightAnalyzer, UndefinedFlyClassifier
import geolocalizacion.weather as weather

# %% INDIVIDUAL BIRDS DESCRIPTION
# =============================================================================
# INDIVIDUAL BIRDS
# =============================================================================
'''
In this initial part, birds are loaded one by one to enrich them and solve
the problems related with the interpolation. Finally the files are saved.
To do it only is necessary to change the name of the bird to load
'''
# %% LOAD FILES
nombre = 'Zorita'
base_path = "E:\\duraton\\geolocalizacion\\_data\\fly"
path_elevation_weather = f"{base_path}\\enriquecida_elevation_weather\\{nombre}_elevation_weather.csv"
df = pd.read_csv(path_elevation_weather,
                 parse_dates = ['UTC_datetime'],
                 index_col=False, 
                 encoding="ISO-8859-1")

# %% INTERPOLATE
df_interpolated = dp.reindex_interpolate(df, freq = 5)
df_interpolated = df_interpolated.drop(labels=['specie', 'name', 'color', 
                                               'flying_situation'], axis=1)
# %% ENRICHMENT
path_raw = f"{base_path}\\raw"
filenames = dp.find_csv_filenames(path_raw)
df_rich = dp.enriquecer(df_interpolated, filenames)
df_rich= df_rich.dropna(subset=['Latitude'])
df_rich['closest_location'] = None

# %% CALCULATE FLYING POSITIONS FOR NON INTERPOLATED DF
plt.close('all')
Fa = FlightAnalyzer(df)
x, freq, n_start, n_end = Fa.get_histogram(column='speed_km_h')
params, cov_matrix = Fa.optimize_parameters(x, freq, plot=False)
uncertain_values = Fa.find_flying_uncertainty(x, freq, params, plot=False)

# %% APPLY FLYING POSITIONS TO INTERPOLATED DF
# plt.close('all')
Fa = FlightAnalyzer(df_rich)
df_rich = Fa.define_flying_situation(uncertain_values)
# df_rich.boxplot('speed_km_h', by='flying_situation')
# %% PREDICT UNDEFINED FLYING VALUES
Ufc = UndefinedFlyClassifier()
df_fly = Ufc.train_model(df_rich)
# df_fly.boxplot('speed_km_h', by='flying_situation')

# %% SAVE INTERPOLATED DF
interpolated_directory = f"{base_path}\\interpolated"
filepath_interpolated = f"{interpolated_directory}\\{nombre}_interpolated.csv"
df_fly.to_csv(filepath_interpolated, index=False, encoding="ISO-8859-1")



# %% ALL BIRDS DESCRIPTION
# =============================================================================
# ALL BIRDS
# =============================================================================
'''
In this final part, interpolated data from all birds is loaded to join it into
one single interpolated file.
'''
# %% LOAD ALL FILES
filenames = dp.find_csv_filenames(interpolated_directory)
filenames = [item for item in filenames if 'all' not in item]
df = dp.load_data(interpolated_directory, filenames, reindex_data=False, pre_process=False)

# %% FILTER
df_end = df.copy()
# Todos los valores entre 0 y -10 m de altura se considerarán = 0
df_end['bird_altitude'] = df_end['bird_altitude'].apply(lambda x: 0 if -10 <= x <= 0 else x)
altitude_condition = (df_end.bird_altitude>=0)
time_step_condition = (df_end.time_step_s>285) & (df_end.time_step_s<315)
satellite_condition = (df_end.satcount>4)
all_conditions = altitude_condition & time_step_condition & satellite_condition 

df_end = df_end[all_conditions]

# %% FILTER ANALYSIS
original_registers = len(df)
final_registers = len(df_end)

print('Registros antes del filtrado: ', original_registers)
print('Registros tras del filtrado: ', final_registers)
porcentaje_desecho = round(100*(original_registers-final_registers)/original_registers, 2)
print(f'Se han desechado un {porcentaje_desecho}% de los datos iniciales')


# %% SAVE ALL FILES UNIFIED
path_all= '\\'.join([interpolated_directory, 'all_interpolated_data.csv'])
df_end.to_csv(path_all, index=False, encoding="ISO-8859-1")

# %% INTERACTING BIRDS
# =============================================================================
# INTERACTING BIRDS
# =============================================================================
names_list = ['Conquista', 'Zorita']
df_interaction = dp.get_same_data(df_end, names_list)
# %% SAVE INTERACTIONS
name_join = '_'.join(names_list)
join_list = [interpolated_directory, 'interactions', f'{name_join}.csv']
filepath_interactions = '\\'.join(join_list)
df_interaction.to_csv(filepath_interactions, index=False, encoding="ISO-8859-1")