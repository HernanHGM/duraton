# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

# OWN LIBRARIES
from sexaje.data_cleaning import Cleaner
from sexaje.data_preprocessing import Preprocessor
from sexaje.model_creation import SelectorClassifier
from sexaje.model_saving import save_models
from sexaje import parameters
import geolocalizacion.data_processing as dp
from geolocalizacion.flying_discrimination import FlightAnalyzer, UndefinedFlyClassifier
# %% LOAD DATA
path= "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather"
path_all= '\\'.join([path, 'all_data.csv'])
df = pd.read_csv(path_all,
                 index_col=False,
                 encoding="ISO-8859-1")

df = df[df.name=='Gato']

# %% DEFINO VUELO Y POSADO
plt.close('all')
Fa = FlightAnalyzer(df)
x, freq, n_start, n_end = Fa.get_histogram(column='speed_km_h')
params, cov_matrix = Fa.optimize_parameters(x, freq, plot=True)
uncertain_values = Fa.find_flying_uncertainty(x, freq, params, plot=True)
df = Fa.define_flying_situation(uncertain_values)
df.boxplot('speed_km_h', by='flying_situation')
# %% PREDICT UNDEFINED FLYING VALUES
Ufc = UndefinedFlyClassifier()
df_2 = Ufc.train_model(df.copy())
df_2.boxplot('speed_km_h', by='flying_situation')
# %% PLOT MAG

plt.close('all')

condition = df['flying_situation'] == 'undefined'
fig1, ax1 = plt.subplots(2,2)
df.loc[condition].hist('mag', bins=100, align='left', ax=ax1[0,0])
df.loc[condition].hist('mag_x', bins=100, align='left', ax=ax1[1,0])
df.loc[condition].hist('mag_y', bins=100, align='left', ax=ax1[0,1])
df.loc[condition].hist('mag_z', bins=100, align='left', ax=ax1[1,1])
fig1.suptitle('indefinido')

condition = df['flying_situation'] == 'landed'
fig2, ax2 = plt.subplots(2,2)
df.loc[condition].hist('mag', bins=100, align='left', ax=ax2[0,0])
df.loc[condition].hist('mag_x', bins=100, align='left', ax=ax2[1,0])
df.loc[condition].hist('mag_y', bins=100, align='left', ax=ax2[0,1])
df.loc[condition].hist('mag_z', bins=100, align='left', ax=ax2[1,1])
fig2.suptitle('posado')

condition = df['flying_situation'] == 'flying'
fig3, ax3 = plt.subplots(2,2)
df.loc[condition].hist('mag', bins=100, align='left', ax=ax3[0,0])
df.loc[condition].hist('mag_x', bins=100, align='left', ax=ax3[1,0])
df.loc[condition].hist('mag_y', bins=100, align='left', ax=ax3[0,1])
df.loc[condition].hist('mag_z', bins=100, align='left', ax=ax3[1,1])
fig3.suptitle('volando')

# %% PLOT ACC

plt.close('all')

condition = df['flying_situation'] == 'undefined'
fig1, ax1 = plt.subplots(2,2)
df.loc[condition].hist('acc', bins=100, align='left', ax=ax1[0,0])
df.loc[condition].hist('acc_x', bins=100, align='left', ax=ax1[1,0])
df.loc[condition].hist('acc_y', bins=100, align='left', ax=ax1[0,1])
df.loc[condition].hist('acc_z', bins=100, align='left', ax=ax1[1,1])
fig1.suptitle('indefinido')

condition = df['flying_situation'] == 'landed'
fig2, ax2 = plt.subplots(2,2)
df.loc[condition].hist('acc', bins=100, align='left', ax=ax2[0,0])
df.loc[condition].hist('acc_x', bins=100, align='left', ax=ax2[1,0])
df.loc[condition].hist('acc_y', bins=100, align='left', ax=ax2[0,1])
df.loc[condition].hist('acc_z', bins=100, align='left', ax=ax2[1,1])
fig2.suptitle('posado')

condition = df['flying_situation'] == 'flying'
fig3, ax3 = plt.subplots(2,2)
df.loc[condition].hist('acc', bins=100, align='left', ax=ax3[0,0])
df.loc[condition].hist('acc_x', bins=100, align='left', ax=ax3[1,0])
df.loc[condition].hist('acc_y', bins=100, align='left', ax=ax3[0,1])
df.loc[condition].hist('acc_z', bins=100, align='left', ax=ax3[1,1])
fig3.suptitle('volando')

# %% LIMPIEZA DE FEATURES
columns =  ['flying_situation',
            'mag_x', 'mag_y', 'mag_z', 'mag', 
            'acc_x', 'acc_y', 'acc_z', 'acc']
df_clean = df.loc[df.flying_situation!='undefined', columns]
# for i ,col in enumerate(columns[1:]):
#     new_col = col+'2'
#     df_clean[new_col] = abs(df_clean[col])
preprocessor = Preprocessor(df_clean)
preprocessor.label_encoder('flying_situation')
preprocessor.feature_scaler()
df_encoded = preprocessor.df_encoded
df_scaled = preprocessor.df_scaled
# %% INFORMACIÓN GENERAL
conteos = df_clean.groupby(['flying_situation']).size()
n_vuelo = conteos['flying']
n_posado= conteos['landed']
conteos_str = f'{n_posado} datos posado; {n_vuelo} datos volando'
print(conteos_str)
cm = df_scaled.corr()
# %%
# variables = ['flying_situation', 'mag_x', 'mag_z', 'acc_y', 'acc_z']
df_train = df_scaled#[variables]


# %% BUSQUEDA MEJOR CLASIFICADOR
SC = SelectorClassifier(df_train, 'flying_situation')

classifier_dict = parameters.classifier_dict
for key, value in classifier_dict.items():
    SC.select_best_classifier({key: value}, n_splits=5)
models_results = SC.results_by_model


# %% BUSQUEDA MEJORES VARIABLES
SC.select_best_features()
features_results = SC.results_by_features

# %% GUARDO MODELO A PRODUCTIVIZAR
# saving_path = 'E:\\duraton\\geolocalizacion\\_results\\model_scaler'
# save_models(df_encoded, 'sexo', conteos_str,
#             features_results, saving_path)

# %% 
a = df.groupby('flying_situation').size()

# %%
df.boxplot('bird_altitude', by='flying_situation')

# %%
condition = df_end['flying_situation'] == 'undefined'
fig1, ax1 = plt.subplots(2,2)
df_end.loc[condition].hist('acc', bins=100, align='left', ax=ax1[0,0])
df_end.loc[condition].hist('acc_x', bins=100, align='left', ax=ax1[1,0])
df_end.loc[condition].hist('acc_y', bins=100, align='left', ax=ax1[0,1])
df_end.loc[condition].hist('acc_z', bins=100, align='left', ax=ax1[1,1])
fig1.suptitle('indefinido')

condition = df['flying_situation'] == 'landed'
fig2, ax2 = plt.subplots(2,2)
df_end.loc[condition].hist('acc', bins=100, align='left', ax=ax2[0,0])
df_end.loc[condition].hist('acc_x', bins=100, align='left', ax=ax2[1,0])
df_end.loc[condition].hist('acc_y', bins=100, align='left', ax=ax2[0,1])
df_end.loc[condition].hist('acc_z', bins=100, align='left', ax=ax2[1,1])
fig2.suptitle('posado')

condition = df['flying_situation'] == 'flying'
fig3, ax3 = plt.subplots(2,2)
df_end.loc[condition].hist('acc', bins=100, align='left', ax=ax3[0,0])
df_end.loc[condition].hist('acc_x', bins=100, align='left', ax=ax3[1,0])
df_end.loc[condition].hist('acc_y', bins=100, align='left', ax=ax3[0,1])
df_end.loc[condition].hist('acc_z', bins=100, align='left', ax=ax3[1,1])
fig3.suptitle('volando')