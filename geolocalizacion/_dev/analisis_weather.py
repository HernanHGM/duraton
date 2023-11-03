# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



plt.rc('font', weight='medium', size=12) # controls default text sizes
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend 
plt.rc('figure', titlesize=15)  # fontsize of the figure title
plt.rc('axes', titlesize=15)     # fontsize of the  title

import sys

path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

import geolocalizacion.data_processing as dp
from geolocalizacion.elevation import ElevationLoader, FlyElevationJoiner
from geolocalizacion.flying_discrimination import FlightAnalyzer


# %% CARGO DATOS DE VUELO, ELEVACION Y METEOROLÓGICO
nombre = 'Gato'
path_enriquecido = f"E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\{nombre}_elevation_weather.csv"

df = pd.read_csv(path_enriquecido, index_col=False)
# %% FILTERS
altitude_condition = (df.bird_altitude>0)
time_step_condition = (df.time_step_s>285) & (df.time_step_s<315)
satellite_condition = (df.satcount>4)
all_conditions = altitude_condition & time_step_condition & satellite_condition 

original_registers = len(df)
df = df[all_conditions]
final_registers = len(df)

print('Registros antes del filtrado: ', original_registers)
print('Registros tras del filtrado: ', final_registers)

# %% datos vuelo landeds vs temperatura

def get_flying_metrics_aggregates(df, var_gby, plot=False):
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
        flying = (df_join[fly_col]=='flying')
        landed = (df_join[fly_col]=='landed')
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
        fly_params_list = [['flying', 'red']
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
                                         
        # df_join[flying].plot.scatter(x=var_gby, y='mean_altitude', ax=axs[0,0],
        #                               color='red', label='flying',
        #                               xlabel=xlabel_dict[var_gby],
        #                               ylabel='mean altitude (m)')
        # df_join[landed].plot.scatter(x=var_gby, y='mean_altitude', ax=axs[0,0],
        #                              color='blue', label='landed',
        #                              xlabel=xlabel_dict[var_gby],
        #                              ylabel='mean altitude (m)')
        
        # df_join[flying].plot.scatter(x=var_gby, y='distance_2D_by_hour',
        #                               ax=axs[1,0], color='red', label='flying',
        #                               xlabel=xlabel_dict[var_gby],
        #                               ylabel='displacement by hour (km/h)')
        # df_join[landed].plot.scatter(x=var_gby, y='distance_2D_by_hour',
        #                              ax=axs[1,0], color='blue', label='landed',
        #                              xlabel=xlabel_dict[var_gby],
        #                              ylabel='displacement by hour (km/h)')
        
        # df_join[flying].plot.scatter(x=var_gby, y='flying_time_percentage', ax=axs[0,1],
        #                              color='red', label='flying',
        #                              xlabel=xlabel_dict[var_gby],
        #                              ylabel='flying time percentage (%)' )
        # df_join[landed].plot.scatter(x=var_gby, y='flying_time_percentage', ax=axs[0,1],
        #                              color='blue', label='landed',
        #                              xlabel=xlabel_dict[var_gby],
        #                              ylabel='flying time percentage (%)')
        
        # df_join[flying].plot.scatter(x=var_gby, y='bird_speed', ax=axs[1,1],
        #                               color='red', label='flying',
        #                               xlabel=xlabel_dict[var_gby],
        #                               ylabel='mean instant speed (km/h)')
        # df_join[landed].plot.scatter(x=var_gby, y='bird_speed', ax=axs[1,1],
        #                              color='blue', label='flying',
        #                              xlabel=xlabel_dict[var_gby],
        #                              ylabel='mean instant speed (km/h)')
        specie = df.specie.unique()[0]
        name = df.name.unique()[0]
        fig.suptitle(f'Especie:{specie}, Nombre:{name}')
        fig.tight_layout()
       
    return df_join



plt.close('all')
variables = ['week_number', 'tempC', 'windspeedKmph', 'pressure', 'humidity', 'precipMM']

results_dict = {}
for var_x in variables:   
    results_dict[var_x] = get_flying_metrics_aggregates(df, var_x, True)

# %% COMPRUEBO SENTIDO DATOS
df.plot.scatter(x = 'tempC', y='windspeedKmph')
a = df.loc[df.vuelo==1,['tempC', 'windspeedKmph']].corr()
# a = dg.loc[dg.vuelo==1,['flying_time', 'flying_time_ratio', 
#                         'total_distance_2D', 'mean_distance_2D', 
#                         'mean_altitude', 'n_samples',
#                         'tempC', 'windspeedKmph']].corr()

# b = dg.loc[dg.vuelo==0,['flying_time', 'flying_time_ratio', 
#                         'total_distance_2D', 'mean_distance_2D', 
#                         'mean_altitude', 'n_samples']].corr()






    