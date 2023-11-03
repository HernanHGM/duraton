# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:51:21 2023

@author: HernanHGM

Apply feature reduction techniques and Gaussian Model Mixture to search for 
two gaussian distributions related to both male and female birds
"""
# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

path_duraton = 'E:\\duraton'
if path_duraton not in sys.path:
    sys.path.append(path_duraton)

# %% LOAD DATA

base_path = "E:\\duraton\\huesos\\sexaje\\_data"
filepath = "\\".join([base_path, 'tabla revisada 2019 definitiva.xlsx'])
df = pd.read_excel(filepath)
df = df[df['Especie']=='Perdicera']
# %% SELECCIONO COLUMNAS
# Filtra las columnas de tipo float
float_columns = df\
        .select_dtypes(include=['float'])\
        .drop(labels='carpo', axis=1)\
        .columns\
        .tolist()
# %% PLOT
plt.close('all')
for col in float_columns:
    df.hist(col)
# %% FEATURE REDUCTION
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA

df_analisis = df[float_columns]

# Inicializa el modelo de Análisis de Factores
n = 3
n_componentes = n  # Define el número deseado de componentes principales
fa = FactorAnalysis(n_components=n_componentes)
pca = PCA(n_components=n_componentes)

# Ajusta el modelo a tus datos
fa.fit(df_analisis)
pca.fit(df_analisis)
# Transforma tus datos originales a las nuevas variables obtenidas por el análisis de factores
df_reducido = pd.DataFrame(pca.transform(df_analisis))  # Cambia los nombres según tu elección de n_componentes

# El DataFrame df_reducido ahora contiene las nuevas variables obtenidas por el análisis de factores.
# %%
# plt.close('all')
for col in range(n):
    df_reducido.hist(col)
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Supongamos que tienes un DataFrame 'df' con tus datos, y que deseas ajustar un GMM.
# Asegúrate de seleccionar las variables relevantes de 'df' para el análisis.
# Por ejemplo, seleccionaremos dos variables 'feature1' y 'feature2'.

# Selecciona las variables de interés.
selected_features = ['coxalL', 'esternon', 'humero']
data = df[selected_features]

# Normaliza tus datos (esto es importante para GMM).
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Ajusta el modelo GMM con dos componentes (asumiendo que tienes dos subgrupos).
n_components = 2
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(data_scaled)

# Predice las etiquetas de los componentes para cada punto de datos.
labels = gmm.predict(data_scaled)

# Añade las etiquetas de componente al DataFrame original.
df['gmm_label'] = labels

# Visualiza los resultados.
plt.figure(figsize=(10, 6))

for label in range(n_components):
    plt.scatter(data[df['gmm_label'] == label]['coxalL'], 
                data[df['gmm_label'] == label]['humero'], 
                label=f'Component {label + 1}')

plt.title('Modelado de Mezclas Gaussianas')
plt.xlabel('coxalL')
plt.ylabel('humero')
plt.legend()
plt.show()

