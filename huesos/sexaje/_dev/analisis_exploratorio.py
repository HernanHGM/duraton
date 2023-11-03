# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:51:21 2023

@author: HernanHGM

Exploratory Analysis to look for sex diferences within bones measures
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
# %% SELECCIONO COLUMNAS

# Filtra las columnas de tipo float
float_columns = df\
        .select_dtypes(include=['float'])\
        .drop(labels='carpo', axis=1)\
        .columns\
        .tolist()

# %% SELECCIONO ESPECIES
specie_count = df.groupby('Especie').size()
min_sample = 40
valid_species = specie_count[specie_count>min_sample]\
    .index\
    .to_list()

df_filtered = df[df.Especie.isin(valid_species)]
# %%
plt.close('all')
for col in float_columns:
    df_filtered.hist(col, by='Especie')
