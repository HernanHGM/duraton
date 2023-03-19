# %% IMPORT LIBRERIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import funciones_pajaros.f as f

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# %% CARGO DATOS

file = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Buho Real\\buhos_estudio_arneses.xlsx'
df_original = pd.read_excel (file)

# %% LIMPIEZA DATOS


label = ['Marcador/a']
variables = ['resto teflón',
             'peso', 
             'izda L',	
             'izda DV',	
             'dcha L',	
             'dcha DV',	
             'ancho ala', 
             'ala d',	
             'ala v',	
             '7º  1ª',	
             'cañón en 7ª',
             'antebrazo',	
             'cola',	
             'rectrix c',
             'envergadura',
             'longitud total',	
             'long pico',	
             'alto pico',	
             'ancho pico',	
             'long cabeza',	
             'ancho cabeza',	
             'clave']

data = f.individual_selection(df_original, 'Búho real', ['adulto', 'joven', 'subadulto'], label, variables)

data_clean = f.remove_outliers(data, label)
data_clean = f.drop_nans(data_clean, 0.3)
data_clean = f.promediado_tarsos(data_clean)

# data_scaled, _, _ = f.scaling_encoding(data_clean, label)
data_aug = f.feature_augmentation(data_clean)
# # Como he multiplicado variables entre ellas, debo volver a escalar
data_scaled, X_scaled, Y_scaled = f.scaling_encoding(data_aug, label)
cm = data_scaled.corr()

# %% SEX ANALYSIS
var_names = list(data_aug.columns)
for i in var_names[1:]:
    data_aug.boxplot(i, by='sexo')
    filename = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\graficas\\' + i + '.png'
    # plt.savefig(filename)
# %% OBSERVER ANALYSIS
var_names = list(data_aug.columns)
df = data_clean.join(df_original['Observador'], how ='left')
df.boxplot('peso', by=['Observador', 'sexo'])
# for i in var_names[1:]:
#     df.boxplot(i, by='sexo', patch_artist=True)
# %% RESTO DE TEFLÓN
# linear regression y--> resto teflón X--> longitud total, L8
from sklearn.linear_model import LinearRegression as LR
label = ['resto teflón']
final_var = ['longitud total', 'L media', 'peso', 'cola']
data_def = data_scaled[final_var + label].copy()
X = data_def[final_var].values
y = np.squeeze(data_def[label].values)

model = LR().fit(X,y)
a = model.score(X,y)
b = model.get_params(deep = True)
print(a)
# %% MODELS
# =============================================================================
# Classifiers
# =============================================================================
RF_C = RandomForestClassifier(n_estimators = 20, max_depth = 2, min_samples_split=10, random_state = 0)
LogReg = LogisticRegression()
SVM = SVC()
LDA = LinearDiscriminantAnalysis()
KNN_C = KNeighborsClassifier(n_neighbors = 10)


# %% CALCULO MODELOS
# Se prueban diferentes modelos probando diferente numero de features
# Los features se seleccionan son Sequential forward selection
final_var = ['long pico', 'alto pico']
data_def = data_scaled[label + final_var].copy()
features = data_def.columns[1:]
X = data_def[final_var].values
y = np.squeeze(data_def[label].values)

most_selected_features, kappa_global, accuracy_global = f.best_features(X, y, features, LogReg, n_splits = 10, n_features = 1)


# %% SAVE VALUES
# n_features = 5
# results = np.zeros((n_features, 5), dtype=object)
# results[:,0] = most_selected_features
# results[:,1:3] = kappa_global
# results[:,3:5] = accuracy_global

