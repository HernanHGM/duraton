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
import geolocalizacion.geolocalizacion as geoloc
from geolocalizacion.flying_discrimination import FlightAnalyzer
# %% CARGA Y LIMPIEZA DATOS
path_enriquecido = "E:\\duraton\\geolocalizacion\\_data\\Gato_enriquecida.csv"
df = pd.read_csv(path_enriquecido)
df['acc'] = np.sqrt(df.acc_x**2 +
                    df.acc_y**2 +
                    df.acc_z**2)

df['mag'] = np.sqrt(df.mag_x**2 +
                    df.mag_y**2 +
                    df.mag_z**2)
# %% carga segundo dataframe para probar
path = "E:\\duraton\\geolocalizacion\\_data"
filenames = geoloc.find_csv_filenames(path)
bird_name = 'Zorita'
filenames = [item for item in filenames if bird_name in item]

info_archivos = list(map(geoloc.extract_info, filenames))
info_pajaros = pd.DataFrame(info_archivos, columns=['especie','ID','nombre'])
info_pajaros['color'] = pd.Series(['green', 'blue', 'purple', 'red', 'orange'])


df = geoloc.load_data(path, filenames, reindex_data=False, speed_limit=1)
df = df.merge(info_pajaros, how = 'left', on = 'ID')


bird_condition = (df.nombre == bird_name)
time_condition = (df.time_step_s < 315) & (df.time_step_s > 285)
satelite_condition =  (df.satcount > 4)
condition =  bird_condition & time_condition & satelite_condition
df = df.loc[condition]
print(f'Se han desechado un {round(100*(len(bird_condition)-len(df))/len(bird_condition), 2)}%\
 de los datos iniciales')
print(f'Se va a trabajar con {len(df)} datos')
# %% DEFINO VUELO Y POSADO
Fa = FlightAnalyzer()
x, freq, n_start, n_end = Fa.get_histogram(df, column='speed_km_h')
params, cov_matrix = Fa.optimize_parameters(x, freq, plot=True)
uncertain_values = Fa.find_flying_uncertainty(x, freq, params, plot=True)
df = Fa.define_flying_situation(df, uncertain_values)
df.boxplot('speed_km_h', by='flying_situation')

# %% PLOT MAG

plt.close('all')

condition = df['flying_situation'] == 'indefinido'
fig1, ax1 = plt.subplots(2,2)
df.loc[condition].hist('mag', bins=100, align='left', ax=ax1[0,0])
df.loc[condition].hist('mag_x', bins=100, align='left', ax=ax1[1,0])
df.loc[condition].hist('mag_y', bins=100, align='left', ax=ax1[0,1])
df.loc[condition].hist('mag_z', bins=100, align='left', ax=ax1[1,1])
fig1.suptitle('indefinido')

condition = df['flying_situation'] == 'posado'
fig2, ax2 = plt.subplots(2,2)
df.loc[condition].hist('mag', bins=100, align='left', ax=ax2[0,0])
df.loc[condition].hist('mag_x', bins=100, align='left', ax=ax2[1,0])
df.loc[condition].hist('mag_y', bins=100, align='left', ax=ax2[0,1])
df.loc[condition].hist('mag_z', bins=100, align='left', ax=ax2[1,1])
fig2.suptitle('posado')

condition = df['flying_situation'] == 'volando'
fig3, ax3 = plt.subplots(2,2)
df.loc[condition].hist('mag', bins=100, align='left', ax=ax3[0,0])
df.loc[condition].hist('mag_x', bins=100, align='left', ax=ax3[1,0])
df.loc[condition].hist('mag_y', bins=100, align='left', ax=ax3[0,1])
df.loc[condition].hist('mag_z', bins=100, align='left', ax=ax3[1,1])
fig3.suptitle('volando')

# %% PLOT ACC

plt.close('all')

condition = df['flying_situation'] == 'indefinido'
fig1, ax1 = plt.subplots(2,2)
df.loc[condition].hist('acc', bins=100, align='left', ax=ax1[0,0])
df.loc[condition].hist('acc_x', bins=100, align='left', ax=ax1[1,0])
df.loc[condition].hist('acc_y', bins=100, align='left', ax=ax1[0,1])
df.loc[condition].hist('acc_z', bins=100, align='left', ax=ax1[1,1])
fig1.suptitle('indefinido')

condition = df['flying_situation'] == 'posado'
fig2, ax2 = plt.subplots(2,2)
df.loc[condition].hist('acc', bins=100, align='left', ax=ax2[0,0])
df.loc[condition].hist('acc_x', bins=100, align='left', ax=ax2[1,0])
df.loc[condition].hist('acc_y', bins=100, align='left', ax=ax2[0,1])
df.loc[condition].hist('acc_z', bins=100, align='left', ax=ax2[1,1])
fig2.suptitle('posado')

condition = df['flying_situation'] == 'volando'
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
df_clean = df.loc[df.flying_situation!='indefinido', columns]
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
n_vuelo = conteos['volando']
n_posado= conteos['posado']
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
