"""
Created on Thu Jan  5 12:50:28 2023

@author: Hernán García Mayoral
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, Point

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from folium.plugins import MarkerCluster, HeatMap
import webbrowser

from math import pi

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
        print(str(ID) + ' calculated')
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
        print('kernel ' + str(key) + ' transformed')
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
                  Name: {row["name"]} <br>
                  ID: {row["ID"]} <br>
                  Specie: {row["specie"]} <br>
        '''
        folium.Marker(location = [row['Latitude'], row['Longitude']], 
                      icon = folium.Icon(color=color),
                      popup = txt).add_to(m)

def add_kernels(gdf, base_map):
    custom_scale = gdf['level'].unique().tolist()
    custom_scale.append(1)
    fill_color = gdf['color_map'].mode()[0]
    specie = gdf['specie'].mode()[0].lower()
    legend_name = 'Probabilidad de aparición ' + specie

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
    
    folium.GeoJsonTooltip(['name', 'specie', 'level', 'start_date', 'end_date']).add_to(cp.geojson)

def add_animated_points (df, m, df_kernels = None):
    if df_kernels is not None:
        for ID, subdf in df_kernels.groupby('specie'):
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
                "popup": row["name"],
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

def plot_map(m: folium.Map, 
             file_map: str="E:\\duraton\\geolocalizacion\\_tests\\map.html"):
    m.save(file_map)
    webbrowser.open(file_map)