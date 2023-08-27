"""
Created on Thu Jan  5 12:50:28 2023

@author: Hernán García Mayoral
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point
import os

from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from folium.plugins import MarkerCluster, HeatMap
import webbrowser

from math import pi
# =============================================================================
# IMPORTACION DATOS
# =============================================================================
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def extract_info(nombre_archivo):
    nombre_limpio = nombre_archivo.replace('.csv', '')
    partes = nombre_limpio.split('_')
    
    conversor_especie={'Aquada': 'Águila imperial',
                        'Aquchr': 'Águila real',
                        'Aqufas': 'Águila perdicera'}
    
    especie = conversor_especie[partes[0]]
    ID = int(partes[1])
    nombre = partes[2]
    return especie, ID, nombre

def load_data(path, filenames, reindex_data = True):
    # join root_path and filenames into full path
    full_filenames = (path + '\\' + name for name in filenames) 
    
    li = []
    date_columns = ['UTC_datetime', 'UTC_date', 'UTC_time']
    for name in full_filenames:
        print(name)
        df = pd.read_csv(name, parse_dates = date_columns)
        df.rename(columns={'device_id': 'ID'}, inplace=True)
        df = remove_outliers(df, columns=['Latitude', 'Longitude'])
        if reindex_data == True:
            df = reindex_interpolate(df, freq = 5)
        li.append(df)
    
    df = pd.concat(li, axis=0, ignore_index=True)
    return df
# =============================================================================
# LIMPIEZA
# =============================================================================
def remove_outliers(data, columns = None):

    data_o = data.copy()
    if columns != None:
        data = data[columns]
        
    lim_df = data.quantile((0.05, 0.95)).transpose()
    lim_df['IQR'] = lim_df[0.95] - lim_df[0.05]
    lim_df['Upper'] = lim_df[0.95] + lim_df['IQR']
    lim_df['Lower'] = lim_df[0.05] - lim_df['IQR']
    lim_df = lim_df.transpose()
    
    up_condition = (data <= lim_df.loc['Upper',:])
    low_condition = (data >= lim_df.loc['Lower',:])
    condition = up_condition & low_condition
    
    data_clean = data_o[condition.all(axis=1)]
    
    return data_clean


def equal_index(df, freq = 5):
    freq_str = str(freq) + 'min'
    df_ = df.set_index('UTC_datetime')
    start_date = df_.index.min()
    rounded_date = start_date - timedelta(minutes=start_date.minute%freq, 
                                          seconds=start_date.second)
    
    df_reindexed = df_.reindex(pd.date_range(start = rounded_date,
                                             end = df_.index.max(),
                                             freq = freq_str))

    df_full = pd.concat([df_, df_reindexed], axis = 0)  
    df_full.sort_index(axis = 0, inplace =True)
    df_full.reset_index(inplace = True)
    df_full.rename(columns = {'index': 'UTC_datetime'}, inplace =True)
    df_full.drop_duplicates(inplace=True)
    return df_full

 
def remove_nulls(df, freq = 5):
    # Get rows where too much columns are null
    threshold = int(60/freq)
    n_nulls = df['ID'].isna().rolling(threshold, center=False).sum()
    # shift -threshold + 2 pq threshold solo borraria el último valor con dato
    # +1 para salvar la última fecha con dato
    # +2 para salvar la última fecha generada (valor nulo)
    forward_condition = (n_nulls.shift(-threshold+2) != threshold)
    # +1 para salvar la ultima fecha sin dato
    backward_condition = (n_nulls.shift(1) != threshold)
    condition1 = forward_condition & backward_condition
    
    # Get rows of the new indexes
    c1 = df.UTC_datetime.dt.minute%freq == 0 # Multiplos de 5
    c2 = df.UTC_datetime.dt.second == 0 # Segundos = 0
    condition2 = c1 & c2

    condition = condition1 & condition2
    return condition

def interpolate_clean(df, condition):
    # Same datetim column
    dates = df.UTC_datetime
    dates = dates[condition]
    # Drop datime columns
    df_ = df.drop(labels = ['UTC_datetime', 'UTC_date', 'UTC_time'], axis = 1)

    df_ = df_.interpolate(method ='linear', limit_direction ='forward')
    df_ = df_[condition]
    df_['UTC_datetime'] = dates
    df_['UTC_date'] = dates.dt.date
    df_['UTC_time'] = dates.dt.time
        
    return df_

def reindex_interpolate(df, freq = 5):
    df2 = equal_index(df) 
    condition = remove_nulls(df2, freq)
    df3 = interpolate_clean(df2, condition)
    
    return df3

def get_same_data(df, lista_nombres):
    
    df2 = df[df['nombre'].isin(lista_nombres)]
    dg = pd.DataFrame(df2.groupby('UTC_datetime')['nombre'].nunique())
    dg.rename(columns={'nombre': 'n_data'}, inplace = True)
    dg.reset_index(inplace =True)
    df2 = df2.merge(dg, on='UTC_datetime', how = 'left')
    df3 = df2[df2['n_data']==len(lista_nombres)]
    
    return df3



# =============================================================================
# CÁLCULOS
# =============================================================================
def circular_area(data, r_plane):
    
    if isinstance(data, list) == True:
        geometry = [Point(xy) for xy in data]  
    elif isinstance(data, pd.DataFrame) == True:
        geometry = [Point(xy) for xy in zip(data.Longitude, data.Latitude)]
    
    # Convertimos datos a formato geopandas
    points = gpd.GeoDataFrame(data, geometry=geometry)
    
    # Crea un buffer de r_plane km alrededor de cada punto
    r_deg = 360*r_plane/(6371*2*pi)
    points_buffer = points.geometry.buffer(r_deg)

    # Crea un GeoDataFrame con los buffers
    buffered_points = gpd.GeoDataFrame(geometry=points_buffer)
    
    n = len(buffered_points)
    names = list(map(lambda x: 'antena ' + str(x+1), range(n)))
    buffered_points['name'] = names   
    return buffered_points

def deg_km(v):
    dif = abs(v)
    min_dif = np.minimum(dif, 360-dif)
    rad_dif = 2*np.pi*min_dif/360
    km_dif = 6370*rad_dif
    return km_dif  

def calculate_distance(df1, df2):
    lat_arr = df1.Latitude - df2.Latitude
    d_lat = deg_km(lat_arr)
    
    long_arr = df1.Longitude - df2.Longitude
    d_long = deg_km(long_arr)
    
    d_alt = (df1.Altitude_m - df2.Altitude_m)/1000
    
    distancias = pd.DataFrame()
    distancias['3D'] = np.sqrt(d_lat**2 + d_long**2 + d_alt**2)
    distancias['2D'] = np.sqrt(d_lat**2 + d_long**2) 
    distancias['altura'] = d_alt
    
    return distancias
# =============================================================================
# SNS KERNELS
# =============================================================================
def get_sns_kernels(levels, gdf, col_name = 'ID', show = False):
    kwargs = {'levels': levels,
      'fill': True,
      'cmap': 'Reds',
      'alpha': 0.9} 
      
    kernels = {}
    for ID, subdf in gdf.groupby(col_name):
        _, ax = plt.subplots(figsize=(20, 8))
        kdeplot = sns.kdeplot(x=subdf['Longitude'], y=subdf['Latitude'], ax=ax, **kwargs)
        kernels[ID] = kdeplot
        print(str(ID) + 'calculated')
    if show == False: 
        plt.close('all')
    
    return kernels

# =============================================================================
# TRANSFORMACIÓN DE KDE DE SNS A POLIGONOS
# =============================================================================
def contour_to_polygons(contour):
    # Create a polygon for the countour
    # First polygon is the main countour, the rest are holes
    for ncp,cp in enumerate(contour.to_polygons()):
        x = cp[:,0]
        y = cp[:,1]
        new_shape = Polygon([(cord[0], cord[1]) for cord in zip(x,y)])
        if ncp == 0:
            poly = new_shape
        else:
            # Remove holes, if any
            poly = poly.difference(new_shape)
    return poly

def poly_to_multipoly(col, level):
    # Cada nivel puede tener varios polígonos que unimos en uno solo
    paths = list(map(contour_to_polygons, col.get_paths()))
    multi = MultiPolygon(paths) 
    return  (level, multi)


def get_gdf_kernels(sns_kernels, levels):
    kernels_list=[]
    for key in sns_kernels:
        level_polygons = list(map(poly_to_multipoly, sns_kernels[key].collections, levels))
        kernel_df = pd.DataFrame(level_polygons, columns =['level', 'geometry'])
        kernel_df['ID'] = key
        kernel_gdf = gpd.GeoDataFrame(kernel_df, geometry='geometry', crs = 'epsg:4326')
        kernels_list.append(kernel_gdf)
        print('kernel' + str(key) + 'transformed')
    gdf_kernels = pd.concat(kernels_list)  
    
    return(gdf_kernels)

def partial_kernels(kde_levels, gdf):
    if 1 not in kde_levels:
        kde_levels.append(1)
    sns_kernels = get_sns_kernels(kde_levels, gdf)
    gdf_kernels = get_gdf_kernels(sns_kernels, kde_levels)
    gdf_kernels['kernel_id'] = (gdf_kernels['ID'].astype(str) + 
                                '_' + 
                                gdf_kernels['level'].astype(str))
    return gdf_kernels 

def total_kernels(kde_levels, gdf):
    if 1 in kde_levels:
        kde_levels = kde_levels[:-1]
    kernels_list = [[level, 1] for level in kde_levels]

    a = []
    for kde in kernels_list:
        a.append(partial_kernels(kde, gdf))
    b = pd.concat(a)
    return b  

# =============================================================================
# DIBUJAR EN MAPA
# =============================================================================
def add_cluster(gdf, col_name, base_map):
    values = gdf[col_name].unique()
    # CLUSTERES
    cluster_dict = {f'cluster_{i}': MarkerCluster().add_to(base_map) for i in values}
    # cluster_list = [ MarkerCluster().add_to(m) for i in range(n_id)]

    # ANTIGUO
    # def add_to_cluster(row):
    #     sim_geo = gpd.GeoSeries(row['geometry'])
    #     geo_j = sim_geo.to_json()
    #     folium.GeoJson(data=geo_j).add_to(cluster_dict[f'cluster_{i}'])
     
    #NUEVO
    def add_to_cluster(row):
        location = [row['geometry'].y, row['geometry'].x]
        popup_html = f'''Longitude: {row["Longitude"]}<br>
                         Latitude: {row["Latitude"]}<br>
                         Datetime: {row["UTC_datetime"]}<br>
                         Speed: {row["speed_km_h"]}'''
        marker = folium.Marker(location=location, popup=popup_html)
        marker.add_to(cluster_dict[f'cluster_{i}'])


    for i, subdf in gdf.groupby(col_name): 
        subdf.apply(add_to_cluster, axis=1)

def add_heatmap(gdf, col_name, base_map):
    for _, subdf in gdf.groupby(col_name): 
        heat_data = [[row['Latitude'],row['Longitude']] for index, row in subdf.iterrows()] 
        # Plot it on the map
        HeatMap(heat_data).add_to(base_map)
        
def add_geometry(gdf, base_map, color):
    def draw_geometry(row, base_map, color='#3388ff'):
        sim_geo = gpd.GeoSeries(row['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                                style_function=lambda x: {'fillColor': color})
        folium.Popup(row['name']).add_to(geo_j)
        geo_j.add_to(base_map)
       
    gdf.apply(lambda row: draw_geometry(row, base_map, color), axis=1)
    
def add_markers(gdf, m):

    # create a color dictionary for the birds
    colors = {211981: 'red', 
              192663: 'blue', 
              213861: 'green', 
              201254: 'purple', 
              201255: 'orange'}

    # iterate through the bird data and add markers to the map
    for _, row in gdf.iterrows():
        color = colors[row['ID']]
        txt = f'''Bird ID: {row["ID"]} <br> 
                  Time: {row["UTC_datetime"]} <br>
                  Nombre: {row["nombre"]} <br>
                  ID: {row["ID"]} <br>
                  Especie: {row["especie"]} <br>
        '''
        folium.Marker(location = [row['Latitude'], row['Longitude']], 
                      icon = folium.Icon(color=color),
                      popup = txt).add_to(m)

def add_kernels(gdf, base_map):
    custom_scale = gdf['level'].unique().tolist()
    custom_scale.append(1)
    fill_color = gdf['color_map'].mode()[0]
    especie = gdf['especie'].mode()[0].lower()
    legend_name = 'Probabilidad de aparición ' + especie

    cp = folium.Choropleth(
        geo_data=gdf,
        name='choropleth',
        data=gdf,
        columns=['kernel_id', 'level'],
        key_on='feature.properties.kernel_id',
        threshold_scale=custom_scale,
        fill_color=fill_color,
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name=legend_name
    ).add_to(base_map) 
    
    folium.GeoJsonTooltip(['nombre', 'especie', 'level', 'start_date', 'end_date']).add_to(cp.geojson)

def add_animated_points (df, m, df_kernels = None):
    if df_kernels is not None:
        for ID, subdf in df_kernels.groupby('especie'):
            add_kernels(subdf, m)
            
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["Longitude"], row["Latitude"]],
            },
            "properties": {
                "time": row["UTC_datetime"].strftime('%Y-%m-%d %H:%M:%S'),
                "popup": row["nombre"],
                'icon': 'circle',
                'iconstyle':{
                       'fillColor': row['color'],
                       'fillOpacity': 1,
                       'stroke': 'true',
                       'radius': row['Altitude_m']/80
                },
            },
        }
        for index, row in df.iterrows()
    ]

    plugins.TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT5M",
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=100,
        loop_button=True,
        date_options="YYYY/MM/DD HH:MM:SS",
        time_slider_drag_update=True,
        duration="PT10M",
    ).add_to(m)

def plot_map(m):
    file_map = "E:\\trabajo_pajaros\\geolocalización\\map.html"
    m.save(file_map)
    webbrowser.open(file_map)