# %% FILE DESCRIPTION
"""
Created on Wed Oct 18 22:51:42 2023

@author: HernanHGM

Load the enriched data from all the birds and join them into one single file
"""
# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)
    
import geolocalizacion.data_processing as dp
# %% LOAD ALL FILES

path= "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather"
filenames = dp.find_csv_filenames(path)
filenames = [item for item in filenames if 'all' not in item]
df = dp.load_data(path, filenames, reindex_data=False, pre_process=False)

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
path_all= '\\'.join([path, 'all_data.csv'])
df_end.to_csv(path_all, index=False, encoding="ISO-8859-1")

