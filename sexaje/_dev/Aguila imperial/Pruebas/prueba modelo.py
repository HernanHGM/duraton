# %% IMPORT LIBRERIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

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

file = 'E:\\trabajo_pajaros\\marcajes\\Tabla_2022_09.xls'
df_original = pd.read_excel (file,sheet_name='Hoja1')

# %% FUNCIONES PREPROCESADO
def individual_selection (df, especie, edad, label, variables):
    condition_especie = (df['especie']==especie)
    condition_edad = (df['edad'].isin(edad)) 
    condition_sexo = (df['sexo'].isin(['macho', 'hembra']))
    
    global_condition = condition_especie & condition_edad & condition_sexo
    
    df_var = df.loc[global_condition, variables].apply(pd.to_numeric) 
    df_label = df.loc[global_condition, label]
    
    df_selected = pd.concat((df_label, df_var), axis = 1)
    
    return df_selected

def promediado_tarsos(df):
    df['L media'] = df.loc[:,['izda L', 'dcha L']].mean(axis=1)
    df['DV media'] = df.loc[:,['izda DV', 'dcha DV']].mean(axis=1)
    df.drop(columns = ['izda L', 'dcha L', 'izda DV', 'dcha DV'], inplace = True)
    return df

def feature_augmentation(df):
          
    if 'L media' in df.columns and 'DV media' in df.columns: 
        df['area tarso'] = df['L media'] * df['DV media']
    
    if 'envergadura' in df.columns and 'longitud total' in df.columns:  
        df['area total'] = df['envergadura'] * df['longitud total']
       
    if 'envergadura' in df.columns and 'ancho ala' in df.columns:  
        df['area ala'] = df['envergadura'] * df['ancho ala']
    
    if 'long pico' in df.columns and 'alto pico' in df.columns:
        df['tamaño pico'] = df['long pico'] * df['alto pico']
    
    if 'long cabeza' in df.columns and 'ancho cabeza' in df.columns:
        df['tamaño cabeza'] = df['long cabeza'] * df['ancho cabeza']
    
    if 'peso' in df.columns and 'antebrazo' in df.columns:
        df['volumen'] = df['peso'] * df['antebrazo']
        
    return df


def remove_outliers(data, *label):
    
    if label:
        Y = data[label[0]]
        data = data.drop(columns = label[0])
        
    lim_df = data.quantile((0.05, 0.95)).transpose()
    lim_df['IQR'] = lim_df[0.95] - lim_df[0.05]
    lim_df['Upper'] = lim_df[0.95] + lim_df['IQR']
    lim_df['Lower'] = lim_df[0.05] - lim_df['IQR']
    lim_df = lim_df.transpose()
    
    up_condition = (data <= lim_df.loc['Upper',:])
    low_condition = (data >= lim_df.loc['Lower',:])
    condition = up_condition & low_condition
    
    data_clean = data[condition]
    
    if label:
        data_clean = pd.concat((Y, data_clean), axis = 1)
    
    
    return(data_clean)

def drop_nans(df, threshold):
    na_proportion = df.isna().sum()/len(df)
    condition = na_proportion < threshold
    df_clean = df.loc[:,condition]
    
    df_clean = df_clean.dropna()
    
    return df_clean

def scaling_encoding(df, label, *variables):
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    
    if variables:
        predictors = df[variables[0]]
        X = pd.DataFrame(MinMaxScaler().fit_transform(predictors), columns = predictors.columns)
    else:
        predictors = df.drop(columns = label) 
        X = pd.DataFrame(MinMaxScaler().fit_transform(predictors), columns = predictors.columns)
    #np.squeeze because fit_transform needs shape (n,)
    Y = pd.DataFrame(LabelEncoder().fit_transform(np.squeeze(df[label])), columns = label)
    data_scaled = pd.concat((Y,X), axis = 1)
    
    return data_scaled, X, Y



label = ['sexo']
variables = ['izda L',	
             'izda DV',	
             'dcha L',	
             'dcha DV']

# %% LIMPIEZA DATOS
data = individual_selection(df_original, 'Buitre negro', ['pollo'], label, variables)

data_clean = remove_outliers(data, label)
data_clean = drop_nans(data_clean, 0.1)
data_clean = promediado_tarsos(data_clean)

# %% SCALING 1
scaler_1 = MinMaxScaler()

x_1 = data_clean[['L media', 'DV media']].values
scaler_1.fit(x_1)

## Prueba
# y = np.array([13, 16]). reshape(1,-1)
# y2 = scaler.transform(y)


filename_1 = 'E:\\trabajo_pajaros\\marcajes\\scaler_1.pkl'
pickle.dump(scaler_1, open(filename_1, 'wb'))

# %% SCALER 2
# Como he multiplicado variables escaladas entre ellas, 
# Las nuevas variables son más pequeñas, por lo que debo volver a escalar
data_scaled, _, _ = scaling_encoding(data_clean, label)
data_aug = feature_augmentation(data_scaled)

scaler_2 = MinMaxScaler()

x_2 = data_aug[['area tarso']].values
scaler_2.fit(x_2)

## Prueba
# y = np.array([0.1]). reshape(1,-1)
# y2 = scaler_2.transform(y)

filename_2 = 'E:\\trabajo_pajaros\\marcajes\\scaler_2.pkl'
pickle.dump(scaler_2, open(filename_2, 'wb'))


# %% MODEL
data_def, X_scaled, Y_scaled = scaling_encoding(data_aug, label)
X = data_def.loc[:,['area tarso']].to_numpy()
Y = data_def['sexo'].to_numpy()

model = LogisticRegression()
model.fit(X, Y)

filename_3 = 'E:\\trabajo_pajaros\\marcajes\\model.pkl'
pickle.dump(model, open(filename_3, 'wb'))
