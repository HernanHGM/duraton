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

import geolocalizacion.weather as weather
import geolocalizacion.geolocalizacion as geoloc
import geolocalizacion.elevation as ele
from geolocalizacion.flying_discrimination import FlightAnalyzer

# %% GUARDO O CARGO DATOS DE VUELO, ELEVACION Y METEOROLÓGICO
path_enriquecido = "E:\\duraton\\geolocalizacion\\_data\\Gato_enriquecida.csv"
df = pd.read_csv(path_enriquecido)
df['acc'] = np.sqrt(df.acc_x**2 +
                                df.acc_y**2 +
                                df.acc_z**2)

df['mag'] = np.sqrt(df.mag_x**2 +
                                df.mag_y**2 +
                                df.mag_z**2)
df['month'] = pd.to_datetime(df['UTC_date'], format="%Y/%m/%d").dt.strftime('%B')
df['week_number'] = pd.to_datetime(df['UTC_date'], format="%Y/%m/%d")\
                      .dt.strftime('%W')\
                      .astype(int)
df = df[(df.time_step_s>285) & (df.time_step_s<315)]

# %% DEFINO VUELO Y POSADO
plt.close('all')
Fa = FlightAnalyzer()
x, freq, n_start, n_end = Fa.get_histogram(df, column='speed_km_h', plot=True)
params, cov_matrix = Fa.optimize_parameters(x, freq, plot=True)
uncertain_values = Fa.find_flying_uncertainty(x, freq, params, plot=True)
df = Fa.define_flying_situation(df, uncertain_values)
df.boxplot('speed_km_h', by='flying_situation')
df.groupby('flying_situation').size()

# %% ANÑALISIS INICIAL DATOS
# Pinto altitudes en funcion de su situacion de vuelo
df.boxplot('bird_altitude', by='flying_situation')
df.boxplot('Altitude_m', by='flying_situation')
df.boxplot('elevation', by='flying_situation')

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
        
        df_join[flying].plot.scatter(x=var_gby, y='mean_altitude', ax=axs[0,0],
                                      color='red', label='flying',
                                      xlabel=xlabel_dict[var_gby],
                                      ylabel='mean altitude (m)')
        df_join[landed].plot.scatter(x=var_gby, y='mean_altitude', ax=axs[0,0],
                                     color='blue', label='landed',
                                     xlabel=xlabel_dict[var_gby],
                                     ylabel='mean altitude (m)')
        
        df_join[flying].plot.scatter(x=var_gby, y='distance_2D_by_hour',
                                      ax=axs[1,0], color='red', label='flying',
                                      xlabel=xlabel_dict[var_gby],
                                      ylabel='displacement by hour (km/h)')
        df_join[landed].plot.scatter(x=var_gby, y='distance_2D_by_hour',
                                     ax=axs[1,0], color='blue', label='landed',
                                     xlabel=xlabel_dict[var_gby],
                                     ylabel='displacement by hour (km/h)')
        
        df_join[flying].plot.scatter(x=var_gby, y='flying_time_percentage', ax=axs[0,1],
                                     color='red', label='flying',
                                     xlabel=xlabel_dict[var_gby],
                                     ylabel='flying time percentage (%)' )
        df_join[landed].plot.scatter(x=var_gby, y='flying_time_percentage', ax=axs[0,1],
                                     color='blue', label='landed',
                                     xlabel=xlabel_dict[var_gby],
                                     ylabel='flying time percentage (%)')
        
        df_join[flying].plot.scatter(x=var_gby, y='bird_speed', ax=axs[1,1],
                                      color='red', label='flying',
                                      xlabel=xlabel_dict[var_gby],
                                      ylabel='mean instant speed (km/h)')
        df_join[landed].plot.scatter(x=var_gby, y='bird_speed', ax=axs[1,1],
                                     color='blue', label='flying',
                                     xlabel=xlabel_dict[var_gby],
                                     ylabel='mean instant speed (km/h)')
        specie = df.especie.unique()[0]
        name = df.nombre.unique()[0]
        fig.suptitle(f'Especie:{specie}, Nombre:{name}')
        fig.tight_layout()
        # im = df_join.plot.scatter(x=var_gby, y='total_distance_2D', c=fly_col,
        #                       cmap='viridis', ax = axs[1,0])
        # im = df_join.plot.scatter(x=var_gby, y='flying_time', c=fly_col,
        #                       cmap='viridis', ax = axs[0,1])
        # im = df_join.plot.scatter(x=var_gby, y='flying_time_ratio', c=fly_col,
        #                       cmap='viridis', ax = axs[1,1])
        # im = df_join.plot.scatter(x=var_gby, y='bird_speed', c=fly_col,
        #                       cmap='viridis', ax = axs[2,1])

        
        # fig2, axs2 = plt.subplots(3,2,figsize=(15,6))
        # im = df_join.plot.scatter(x=var_gby, y='mean_altitude', 
        #                           s='n_samples', c=fly_col, 
        #                       cmap='viridis', ax = axs2[0,0])
        # im = df_join.plot.scatter(x=var_gby, y='total_distance_2D', 
        #                           s='n_samples', c=fly_col,
        #                       cmap='viridis', ax = axs2[1,0])
        # im = df_join.plot.scatter(x=var_gby, y='mean_distance_2D', 
        #                           s='n_samples', c=fly_col,
        #                       cmap='viridis', ax = axs2[2,0])
        # im = df_join.plot.scatter(x=var_gby, y='flying_time', 
        #                           s='n_samples', c=fly_col,
        #                       cmap='viridis', ax = axs2[0,1])
        # im = df_join.plot.scatter(x=var_gby, y='flying_time_ratio', 
        #                           s='n_samples', c=fly_col,
        #                       cmap='viridis', ax = axs2[1,1])
        # im = df_join.plot.scatter(x=var_gby, y='bird_speed', 
        #                           s='n_samples', c=fly_col,
        #                       cmap='viridis', ax = axs2[2,1])

        
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






    