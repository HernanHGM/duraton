# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:33:50 2024

@author: Usuario
"""
import pandas as pd

path_interpolated_groupby = "E:\\duraton\\geolocalizacion\\_data\\fly\\interpolated\\all_interpolated_grouped_data.csv"
df_interpolated_groupby = pd.read_csv(path_interpolated_groupby,
                                index_col=False, 
                                encoding="ISO-8859-1")
# %%
path_interpolated = "E:/duraton/geolocalizacion/_data/fly/interpolated/all_interpolated_data.csv"
df_interpolated = pd.read_csv(path_interpolated,
                                index_col=False, 
                                encoding="ISO-8859-1")
# %%
path_all_data= "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\all_data.csv"
df_all_data = pd.read_csv(path_all_data,
                                index_col=False, 
                                encoding="ISO-8859-1")

# %%
path_all_grouped = "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\all_grouped_data.csv"
df_all_grouped = pd.read_csv(path_all_grouped,
                                index_col=False, 
                                encoding="ISO-8859-1")

# %% 
interpolated_columns = list(df_interpolated.columns)

interpolated_columns_gby = list(df_interpolated_groupby.columns)
normal_columns = list(df_all_data.columns)

all_grouped_columns_gby = list(df_all_grouped.columns)

main_list = list(set(interpolated_columns) - set(interpolated_columns_gby))
main_list2 = list(set(interpolated_columns) - set(normal_columns))

a = ('breeding_period' in main_list)

# %%
a = df_all_data[df_all_data.name=='Conquista']
b = df_interpolated[df_interpolated.name=='Conquista']

# %%
nombre = 'Conquista'
base_path = "E:\\duraton\\geolocalizacion\\_data\\fly"
path_elevation_weather = f"{base_path}\\enriquecida_elevation_weather\\{nombre}_elevation_weather.csv"
df = pd.read_csv(path_elevation_weather,
                 parse_dates = ['UTC_datetime'],
                 index_col=False, 
                 encoding="ISO-8859-1")
# %%
import sys
path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)
    
import geolocalizacion.data_processing as dp
path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw"
filenames = dp.find_csv_filenames(path)
nombre = 'Conquista'
filenames = [item for item in filenames if nombre in item]
 
## %% IMPORT BIRDS DATA
df_fly = dp.load_data(path, filenames, reindex_data=False)
