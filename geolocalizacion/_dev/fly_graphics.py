# %% DESCRIPTION
"""
Created on Thu Oct 19 23:35:54 2023

@author: HernanHGM

Once the finel file is prepared, create grapichs
"""
# %% IMPORT PANDAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %% LOAD PREPARED FILE
df = pd.read_csv("E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\all_data.csv",
                               index_col=False, 
                               encoding="ISO-8859-1")

# %% AGGREGATE_FUNCTION


def get_flying_metrics_aggregates(df, 
                                  var_gby, 
                                  plot=False, 
                                  save_path='E:\\duraton\\geolocalizacion\\_results\\fly_weather'):
    fly_col = 'flying_situation'
    df_general_agg = df.groupby([fly_col, var_gby])\
                       .agg(mean_altitude=('bird_altitude', 'mean'),
                            flying_time=('time_step_s', 'sum'),
                            total_distance_2D=('distance_2D', 'sum'),
                            bird_speed=('speed_km_h', 'mean'),
                            n_samples=('distance_2D', 'count'))\
                       .reset_index()
                       
    df_time_agg = df.groupby([var_gby])\
                    .agg(total_time=('time_step_s', 'sum'))\
                    .reset_index()
               
    df_join = pd.merge(df_general_agg, df_time_agg, on=var_gby)
    df_join['flying_time_percentage'] = 100*df_join['flying_time']/df_join['total_time']
    df_join['distance_2D_by_hour'] = 3600*df_join['total_distance_2D']/df_join['flying_time']

    if plot == True:
        xlabel_dict = {'week_number': 'Week of the year',
                       'tempC': 'Temperature (ºC)',
                       'windspeedKmph': 'Wind speed (km/h)',
                       'pressure': 'Atmospheric pressure (mbar)',
                       'humidity': 'realtive humidity (%)',
                       'precipMM': 'Precipitation (mm)'}
        
        fig, axs = plt.subplots(2,2,figsize=(10,6))
        
        yparams_list = [['mean_altitude', axs[0, 0], 'mean altitude (m)'],
                       ['distance_2D_by_hour', axs[1, 0], 'displacement by hour (km/h)'],
                       ['flying_time_percentage', axs[0, 1], 'flying time percentage (%)'],
                       ['bird_speed', axs[1, 1], 'mean instant speed (km/h)']]
        fly_params_list = [['flying', 'red'],
                           ['landed', 'blue']]
        for yparams in yparams_list:
            for fly_params in fly_params_list:
                condition = (df_join[fly_col]==fly_params[0])
                df_join[condition].plot.scatter(x=var_gby,
                                                y=yparams[0], 
                                                ax=yparams[1], 
                                                color=fly_params[1], 
                                                label=fly_params[0],
                                                xlabel=xlabel_dict[var_gby],
                                                ylabel=yparams[2])
                
        specie = df.specie.unique()[0]
        name = df.name.unique()[0]
        fig.suptitle(f'Especie:{specie}, Nombre:{name}')
        fig.tight_layout()
        filename = '\\'.join([save_path, f'{name}_{var_x}.png'])
        plt.savefig(filename)
    return df_join

# %% WEATHER GRAPHICS

plt.close('all')
variables = ['week_number', 'tempC', 'windspeedKmph', 'pressure', 'humidity', 'precipMM']

results_dict = {}
for nombre in df.name.unique():
    print(nombre)
    print('size: ', len(df[df.name==nombre]))
    for var_x in variables:   
        results_dict[var_x] = get_flying_metrics_aggregates(df[df.name==nombre], var_x, True)
# %% RANDOM COLUMN
df['random'] = np.random.randint(1, 1000, df.shape[0])
# %% SPECIE GRAPHICS

fly_col = 'flying_situation'
var_gby = ['random', 'specie', 'name']
df_general_agg = df.groupby([fly_col, *var_gby])\
                   .agg(mean_altitude=('bird_altitude', 'mean'),
                        flying_time=('time_step_s', 'sum'),
                        total_distance_2D=('distance_2D', 'sum'),
                        bird_speed=('speed_km_h', 'mean'),
                        n_samples=('distance_2D', 'count'))\
                   .reset_index()
                   
df_time_agg = df.groupby(var_gby)\
                .agg(total_time=('time_step_s', 'sum'))\
                .reset_index()
           
df_join = pd.merge(df_general_agg, df_time_agg, on=var_gby)
df_join['flying_time_percentage'] = 100*df_join['flying_time']/df_join['total_time']
df_join['distance_2D_by_hour'] = 3600*df_join['total_distance_2D']/df_join['flying_time']


# df_join = get_flying_metrics_aggregates(df, ['UTC_date', 'specie', 'name'], False)
df_join = df_join[df_join.flying_situation=='flying']



# %%
plt.close('all')
df_join.boxplot('flying_time_percentage', by='name')
df_join.boxplot('distance_2D_by_hour', by='name')
df_join.boxplot('mean_altitude', by='name')

df_join.boxplot('bird_speed', by='name')