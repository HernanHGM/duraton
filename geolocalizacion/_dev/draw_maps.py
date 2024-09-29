# %% DESCRIPTION
"""
Created on Wed Feb 21 11:10:59 2024

@author: HernanHGM

To see where the birds are, in this file their kernels are calculated and 
plotted.
"""

# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from shapely.geometry import Point

import sys

path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

import geolocalizacion.data_processing as dp
import geolocalizacion.map_figures as mapf
# %% LOAD DEFINITIVE DATA

all_data = "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\all_data.csv"
df_all = pd.read_csv(all_data,
                     index_col=False, 
                     encoding="ISO-8859-1")
# %% Preprocess
df_draw = df_all[~df_all['name'].isin(['Calizo', 
                                       'Imperial-181672', 
                                       'Imperial-1N',
                                       'Imperial-3X',
                                       'Villamiel'])]

dg = df_draw.groupby(['ID']).agg(start_date=('UTC_date', np.min), end_date=('UTC_date', np.max))
birds_info_path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw\\birds_info.xlsx"
df_birds_info = pd.read_excel(birds_info_path)

# %% ADD COLOR BY SPECIE
geometry = [Point(xy) for xy in zip(df_draw.Longitude, df_draw.Latitude)]
gdf = gpd.GeoDataFrame(df_draw, geometry=geometry, crs='epsg:4326')

def _add_specie_color(row : pd.DataFrame):
    if row['specie'] == 'Aquila fasciata':
        color = 'OrRd'     
    elif row['specie'] == 'Aquila adalberti':
        color = 'YlGn'
    elif row['specie'] == 'Aquila chrysaetos':
        color = 'BuPu'
        
    return color


# %% PLOT BASE MAP WITH FOLIUM
# Creo mapa
base_map = folium.Map(location=[39.6,	-6.42], zoom_start=12)

col_name = 'ID'
# mapf.plot_map(base_map)
# %% CALCULATE KERNELS FOR EACH INDIVIDUAL
# Create thresholds
kde_levels = [0.05, 0.5, 0.75]

gdf_kernels = mapf.partial_kernels(kde_levels, gdf)
# %% ADD EXTRA INFORMATION
# Already exists in definitive data
gdf_kernels = gdf_kernels.merge(df_birds_info, how = 'left', on = 'ID')
gdf_kernels['color_map'] = gdf_kernels[['specie']].apply(_add_specie_color, axis = 1)

gdf_kernels = gdf_kernels.merge(dg, how = 'left', on = 'ID')

gdf_kernels['start_date'] = pd.to_datetime(gdf_kernels['start_date']).dt.strftime('%Y-%m-%d')
gdf_kernels['end_date'] = pd.to_datetime(gdf_kernels['end_date']).dt.strftime('%Y-%m-%d')

# %% PLOT KERNELS OVER BASE MAP
for ID, subdf in gdf_kernels.groupby('specie'):
    mapf.add_kernels(subdf, base_map)
mapf.plot_map(base_map)

# %% SAVE FILE
gdf_kernels_095 = gdf_kernels[gdf_kernels.level==0.05]
gdf_kernels_095.to_file("E:\\duraton\\geolocalizacion\\_results\\kernels\\ultimos datos\\kernels_095_extremadura.shp") 
gdf_kernels.to_file("E:\\duraton\\geolocalizacion\\_results\\kernels\\ultimos datos\\kernels_extremadura.shp")  

# %% INITIAL DATA

'''
This was used to draw the initial data, but the process is repeated with 
the final data, so  this is kind of useless
'''
# # %% DEFINE PATHS
# path_folder = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw"
# birds_info_path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw\\birds_info.xlsx"
# filenames = dp.find_csv_filenames(path)
# # names_list = [e for e in names_list if e not in ('5674')] 
 
# # %% IMPORT BIRDS DATA
# df_fly = dp.load_preprocess_data(path_folder, filenames, origin='movebank', reindex_data=False)
# # df_fly['speed_km_h'].hist(bins=int(df_fly['speed_km_h'].max()-df_fly['speed_km_h'].min())+1)
# dg2 = df_fly.groupby(['name']).agg(start_date=('UTC_date', np.min), end_date=('UTC_date', np.max))
# # LOAD BIRDS individual data
# df_birds_info = pd.read_excel(birds_info_path)


# # %% FILTER DATA
# df_draw = df_fly.copy()

# # Todos los valores entre 0 y -10 m de altura se considerarán = 0
# df_draw['Altitude_m'] = df_draw['Altitude_m'].apply(lambda x: 0 if -10 <= x <= 0 else x)
# altitude_condition = (df_draw.Altitude_m>=0)
# time_step_condition = (df_draw.time_step_s>285) & (df_draw.time_step_s<315)
# satellite_condition = (df_draw.satcount>4)
# all_conditions = altitude_condition & time_step_condition & satellite_condition 

# df_draw = df_draw[all_conditions]
# dg = df_draw.groupby(['ID']).agg(start_date=('UTC_date', np.min), end_date=('UTC_date', np.max))