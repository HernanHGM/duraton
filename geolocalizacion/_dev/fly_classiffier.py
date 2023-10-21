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
from geolocalizacion.flying_discrimination import FlightAnalyzer
# %% LOAD DATA
path= "E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather"
path_all= '\\'.join([path, 'all_data.csv'])
df = pd.read_csv(path_all,
                 index_col=False,
                 encoding="ISO-8859-1")

df = df[df.name=='Gato']

# %% DEFINO VUELO Y POSADO
Fa = FlightAnalyzer(df)
x, freq, n_start, n_end = Fa.get_histogram(df, column='speed_km_h')
params, cov_matrix = Fa.optimize_parameters(x, freq, plot=True)
uncertain_values = Fa.find_flying_uncertainty(x, freq, params, plot=True)
df = Fa.define_flying_situation(df, uncertain_values)
df.boxplot('speed_km_h', by='flying_situation')

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

# %% ENTRENO MODELO

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Supongamos que tienes un DataFrame llamado 'df' con las columnas 'flying_situation' y 'acc_y'.

# Filtra las filas que contienen 'volando' o 'posado' en 'flying_situation'.
filtered_df = df[df['flying_situation'].isin(['flying', 'landed'])]

# Divide el DataFrame en características (X) y etiquetas (y).
X = filtered_df[['acc_y']]
y = filtered_df['flying_situation']

# Divide los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Inicializar el escalador MinMaxScaler.
scaler = MinMaxScaler()

# Ajustar y transformar los datos de entrenamiento.
X_train_scaled = scaler.fit_transform(X_train)

# Transformar los datos de prueba utilizando el mismo escalador.
X_test_scaled = scaler.transform(X_test)


# Crea un clasificador de árbol de decisión.
classifier = DecisionTreeClassifier(random_state=42)

# Entrena el clasificador en los datos de entrenamiento.
classifier.fit(X_train_scaled, y_train)

# Realiza predicciones en el conjunto de prueba.
y_pred = classifier.predict(X_test_scaled)

# Calcula la precisión del clasificador.
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del clasificador: {accuracy:.2f}')

# %%
# Ahora puedes utilizar el clasificador entrenado para predecir 'flying_situation' en nuevos datos.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Supongo que estás usando un clasificador RandomForest, ajusta según el modelo que hayas entrenado.
df_end = df.copy()
# Asegúrate de que 'classifier_model' ya está entrenado con tus datos 'posado' y 'volando'.

# Supongamos que tienes un DataFrame 'df' con la columna 'flying_situation' y 'acc_y'.
# Quieres actualizar la columna 'flying_situation' para aquellos valores en los que es 'indefinido'.

# Filtra los datos donde 'flying_situation' es 'indefinido'.
indefinido_data = df_end[df_end['flying_situation'] == 'undefined']

# Luego, utiliza el modelo para clasificar estos datos y asignar la etiqueta correspondiente.
X = indefinido_data[['acc_y']]  # El conjunto de características es 'acc_y'.

# Ajusta el escalador a los datos de entrenamiento (puede haberlo hecho previamente) o proporciona datos específicos para el ajuste.
scaler.fit(X)

# Transforma los datos de 'indefinido_data' utilizando el escalador.
X_scaled = scaler.transform(X)

predicted_labels = classifier.predict(X_scaled)  # Utiliza el modelo para predecir las etiquetas.

# Asigna las etiquetas predichas al DataFrame original 'df'.
df_end.loc[df_end['flying_situation'] == 'undefined', 'flying_situation'] = predicted_labels

# Ahora, los datos en 'flying_situation' han sido actualizados según las predicciones del modelo.

# 'predicted_labels' contiene las etiquetas ('posado' o 'volando') predichas por el modelo para los datos 'indefinidos'.

# Asegúrate de que 'classifier_model' esté cargado y listo para hacer predicciones.

# Si 'predicted_labels' es una matriz NumPy, puedes convertirla a una Serie de Pandas y asignarla a 'df':
# df.loc[df['flying_situation'] == 'indefinido', 'flying_situation'] = pd.Series(predicted_labels)

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