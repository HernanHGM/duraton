# %% IMPORT LIBRERIAS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# %% CARGA Y LIMPIEZA DATOS
file = 'E:\\duraton\\sexaje\\_data\\Tabla_2022_09.xlsx'

cleaner = Cleaner(file)
cleaner.select_columns()
edad = ['adulto', 'joven', 'subadulto']
especie = 'Águila real'
sexo = ['macho', 'hembra']
cleaner.select_rows(edad, especie, sexo)
cleaner.calculate_means()
cleaner.remove_empty_columns(0.05)
cleaner.remove_outliers()
df_clean = cleaner.df
# %% LIMPIEZA DE FEATURES
categorias = ['edad', 'especie']
features = [x for x in df_clean.columns if x not in categorias]
preprocessor = Preprocessor(df_clean.loc[:, features])
preprocessor.label_encoder('sexo')
scaling_type = 'z_score' #z_score min_max
preprocessor.feature_scaler(scaling_type)
df_encoded = preprocessor.df_encoded
df_scaled = preprocessor.df_scaled
# %% INFORMACIÓN GENERAL
conteos = df_clean.groupby(['sexo']).size()
n_machos = conteos['macho']
n_hembras = conteos['hembra']

conteos_str = f'Training sample: {n_hembras} females; {n_machos} males'
cm = df_scaled.corr()

# %% BUSQUEDA MEJOR CLASIFICADOR
SC = SelectorClassifier(df_scaled, 'sexo')

classifier_dict = parameters.classifier_dict
for key, value in classifier_dict.items():
    SC.select_best_classifier({key: value})
models_results = SC.results_by_model
# Si quisiera elegir otro, quito el que no quiero
# # SC.results_by_model = models_results.drop('c')

# %% BUSQUEDA MEJORES VARIABLES

SC.select_best_features()
features_results = SC.results_by_features
# Uso el peso porque es medida facil de tomar y la clasificacion es mas robusta
features_results = features_results.drop('L_media')

# %% GUARDO MODELO A PRODUCTIVIZAR
saving_path = 'E:\\duraton\\sexaje\\_dev\\Aguila real\\model_scaler'
save_models(df_encoded, 'sexo', conteos_str,
            features_results, saving_path, scaling_type)

# %% GRÁFICAS


groups = df_clean.groupby('sexo')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.L_media, group.cola, marker='o', linestyle='', ms=12, label=name)
ax.legend()

plt.show()

# plt.close('all')
# # categorias = ['sexo', 'edad', 'especie']
# # features = [x for x in df_clean.columns if x not in categorias]
# # for i in features:
# #     df_clean.boxplot(i, by='sexo')
    
# import matplotlib.pyplot as plt
# import scipy.stats as stats
# import numpy as np

# categorias = ['sexo', 'edad', 'especie']
# features = [x for x in df_clean.columns if x not in categorias]

# for i in features:
#     df_clean.boxplot(i, by='sexo')
#     groups = df_clean.groupby('sexo')[i].apply(list)
    
#     # Interpolar las distribuciones para tener el mismo número de valores
#     max_len = max(len(groups[0]), len(groups[1]))
#     interpolated_groups = [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(group)), group) for group in groups]
    
#     # Calcula la distancia de Jensen-Shannon
#     p = np.vstack(interpolated_groups)
#     p = p / np.sum(p, axis=1)[:, np.newaxis]
#     m = 0.5 * (p[0] + p[1])
#     distance_js = 0.5 * (stats.entropy(p[0], m) + stats.entropy(p[1], m))

#     # Calcula la distancia de Kolmogorov-Smirnov
#     n1 = len(interpolated_groups[0])
#     n2 = len(interpolated_groups[1])
#     distance_ks = np.abs(stats.ks_2samp(interpolated_groups[0], interpolated_groups[1])[0]) * np.sqrt((n1 + n2) / (n1 * n2))

#     # Calcula la divergencia de Kullback-Leibler
#     distance_kl = stats.entropy(p[0], p[1])

#     # Anota las métricas de distancia en la gráfica
#     plt.annotate(f'JS distance: {distance_js:.4f}\nKS distance: {distance_ks:.4f}\nKL divergence: {distance_kl:.4f}',
#                  xy=(1, 1), xycoords='axes fraction',
#                  xytext=(-5, -5), textcoords='offset points',
#                  ha='right', va='top', fontsize=10)

#     plt.title("")  # Elimina el título generado por boxplot()
#     plt.xlabel("Sexo")
#     plt.ylabel(i)
#     plt.show()

