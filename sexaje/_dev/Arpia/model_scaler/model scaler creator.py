# %% IMPORT LIBRERIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import funciones_pajaros.f as f

import warnings
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# %% CARGO DATOS

file = 'E:\\trabajo_pajaros\\marcajes\\arpias.xlsx'
df_original = pd.read_excel (file)

# %% LIMPIEZA DATOS
label = ['sexo']
variable = 'cola'
variables = [variable]
data = f.individual_selection(df_original, 'Harpía', ['adulto', 'joven', 'subadulto'], label, variable)

data_clean = f.remove_outliers(data, label)
data_clean = f.drop_nans(data_clean, 0)

# CREATING SCALER & MODEL 
f.scaler_model_creator(data_clean, label, variable)
# %% SCALER augmented
# # Como he multiplicado variables escaladas entre ellas, 
# # Las nuevas variables son más pequeñas, por lo que debo volver a escalar
# data_scaled, _, _ = scaling_encoding(data_clean, label)
# data_aug = feature_augmentation(data_scaled)

# scaler_2 = MinMaxScaler()

# x_2 = data_aug[['area tarso']].values
# scaler_2.fit(x_2)

# ## Prueba
# # y = np.array([0.1]). reshape(1,-1)
# # y2 = scaler_2.transform(y)

# filename_2 = 'E:\\trabajo_pajaros\\marcajes\\scaler_2.pkl'
# pickle.dump(scaler_2, open(filename_2, 'wb'))


