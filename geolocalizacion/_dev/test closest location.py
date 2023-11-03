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
from geopy.distance import geodesic

# %% IMPORT BIRDS DATA

path = "E:\\duraton\\geolocalizacion\\_data\\fly\\raw"
filenames = dp.find_csv_filenames(path)
nombre = 'Deleitosa'
filenames = [item for item in filenames if nombre in item]

df = dp.load_data(path, filenames, reindex_data=False)
# %% IMPORT WEATHER DATA
weather_dict = weather.load_weather_dataframe()
df_locations = weather_dict['coordinates']
# %% FIND CLOSEST LOCATION (Takes ~1min/30K rows)


def find_nearest_location(row, locations_df):
    location_coords = list(zip(locations_df['Latitude'], 
                                locations_df['Longitude']))
    distances = [geodesic((row['Latitude'], row['Longitude']), loc).kilometers\
                  for loc in location_coords]
    nearest_index = distances.index(min(distances))
    closest_location = locations_df.index.tolist()[nearest_index]
    return closest_location

df['closest_location'] = df.apply(find_nearest_location, 
                                    args=(df_locations,), 
                                    axis=1)
# %% PANDAS TO GEOPANDAS
from shapely.geometry import Point
import geopandas as gpd
df2 = df.loc[:1000].copy()
geometry = [Point(xy) for xy in zip(df2.Longitude, df2.Latitude)]
gdf = gpd.GeoDataFrame(df2, geometry=geometry, crs='epsg:4326')

# %% PLOT POINTS MAP
import folium
def add_markers(gdf: gpd.GeoDataFrame, 
                m: folium.Map=folium.Map(location=[39.742117, -5.7012035], zoom_start=12)):

    # create a color dictionary for the locations
    colors_dict = {'Romangordo': 'red', 
              'Deleitosa': 'blue'}

    # iterate through the bird data and add markers to the map
    for _, row in gdf.iterrows():
        color = colors_dict[row['closest_location']]
        txt = f'''Bird ID: {row["ID"]} <br> 
                  Time: {row["UTC_datetime"]} <br>
                  Nombre: {row["name"]} <br>
                  ID: {row["ID"]} <br>
                  Especie: {row["specie"]} <br>
        '''
        folium.Marker(location = [row['Latitude'], row['Longitude']], 
                      icon = folium.Icon(color=color),
                      popup = txt).add_to(m)
    return m

# %%
import time
from geolocalizacion.map_figures import plot_map
start = time.process_time_ns()/(10**9)
# base_map = folium.Map(location=[39.742117, -5.7012035], zoom_start=12)

m = add_markers(gdf)
plot_map(m)
print('Tiempo transcurrido = ', time.process_time_ns() /
      (10**9) - start, ' segundos')