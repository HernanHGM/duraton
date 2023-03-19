# %% IMPORT LIBRERIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
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

def feature_augmentation(df):
    
    df['L media'] = df.loc[:,['izda L', 'dcha L']].mean(axis=1)
    df['DV media'] = df.loc[:,['izda DV', 'dcha DV']].mean(axis=1)
    df.drop(columns = ['izda L', 'dcha L', 'izda DV', 'dcha DV'], inplace = True)
      
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

def drop_nan_columns(df, threshold):
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
variables = ['peso', 
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

# %% LIMPIEZA DATOS
data = individual_selection(df_original, 'Buitre negro', ['pollo'], label, variables)

data_clean = remove_outliers(data, label)
data_clean = drop_nan_columns(data_clean, 0.1)

data_scaled, _, _ = scaling_encoding(data_clean, label)
data_aug = feature_augmentation(data_scaled)
# Como he multiplicado variables entre ellas, debo volver a escalar
data_scaled, X_scaled, Y_scaled = scaling_encoding(data_aug, label)
cm = data_scaled.corr()

# %% selector_and_predictor
def selector_and_predictor(X, Y, model, model_selector, n_splits = 10):
    import time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
    
    start = time.process_time()
    kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    kappa = []
    accuracy = []
    cm = []
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        # X_train_norm, var_mean, var_std = normalize(X_train)
        # X_test_norm = normalize(X_test, var_mean=var_mean, var_std=var_std)
        
        model_selector.fit(X_train, Y_train)
        X_train_reduced = model_selector.transform(X_train)
        X_test_reduced = model_selector.transform(X_test)
        
        model.fit(X_train_reduced, Y_train)
        Y_pred = model.predict(X_test_reduced)
        Y_pred_round = np.round(Y_pred)
        kappa.append( cohen_kappa_score(Y_test, Y_pred_round))
        accuracy.append(accuracy_score(Y_test, Y_pred_round))
        
    print('Tiempo transcurrido = ', time.process_time() - start, ' segundos')
    print('kappa = ', np.mean(kappa).round(2), u"\u00B1", np.std(kappa).round(2))
    print('accuracy = ', np.mean(accuracy).round(2), u"\u00B1", np.std(accuracy).round(2))
# %% MODELS
# =============================================================================
# Classifiers
# =============================================================================
RF_C = RandomForestClassifier(n_estimators = 20, max_depth = 2, min_samples_split=10, random_state = 0)
LogReg = LogisticRegression()
SVM = SVC()
LDA = LinearDiscriminantAnalysis()
KNN_C = KNeighborsClassifier(n_neighbors = 10)

# %%
classifier = LogReg
sfs = SequentialFeatureSelector(classifier, n_features_to_select=2)

X = X_scaled.values
y = np.squeeze(Y_scaled.values)

# selector_and_predictor(X, y, classifier, sfs, n_splits = 10)
sfs.fit(X, y)

# %% DESARROLLO
def best_features(X, Y, features, model, n_splits = 10, n_features = 5):
    import time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
    from statistics import mode
    
    start = time.process_time()
    kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    
    most_selected_features = []
    kappa_global = []
    accuracy_global = []
    for i in range(1, n_features+1):
        
        sfs = SequentialFeatureSelector(model, n_features_to_select = i)
        kappa = []
        accuracy = []
        selected_features = []
        for train_index, test_index in kf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            sfs.fit(X_train, Y_train)
            X_train_reduced = sfs.transform(X_train)
            X_test_reduced = sfs.transform(X_test)
            selected_features.append(tuple(features[sfs.get_support()]))
            
            model.fit(X_train_reduced, Y_train)
            Y_pred = model.predict(X_test_reduced)
            Y_pred_round = np.round(Y_pred)
            
            kappa.append(cohen_kappa_score(Y_test, Y_pred_round))
            accuracy.append(accuracy_score(Y_test, Y_pred_round))
    
        print('Tiempo transcurrido = ', time.process_time() - start, ' segundos')
        print('kappa = ', np.mean(kappa).round(2), u"\u00B1", np.std(kappa).round(2))
        print('accuracy = ', np.mean(accuracy).round(2), u"\u00B1", np.std(accuracy).round(2))
        
        most_selected_features.append(mode(selected_features))
        kappa_global.append([np.mean(kappa).round(2), np.std(kappa).round(2)])
        accuracy_global.append([np.mean(accuracy).round(2), np.std(accuracy).round(2)])
     

    return most_selected_features, kappa_global, accuracy_global

features = data_scaled.columns[1:]
most_selected_features, kappa_global, accuracy_global = best_features(X, y, features, SVM, n_splits = 10, n_features = 5)


# %%
n_features = 5
results = np.zeros((n_features, 5), dtype=object)
results[:,0] = most_selected_features
results[:,1:3] = kappa_global
results[:,3:5] = accuracy_global