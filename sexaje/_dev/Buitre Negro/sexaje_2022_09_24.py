import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import pointbiserialr
import statsmodels.formula.api as smf

file = 'E:\\trabajo_pajaros\\marcajes\\Tabla_2022_09_clean.xls'
df = pd.read_excel (file,sheet_name='Hoja1')
# %% LIMPIEZA
especies = df['especie'].unique()

index_imperial = df['especie'].isin(['imperial', 'Imperial', 'Águila imperial', 'Aguila imperial'])
index_perdicera = df['especie'].isin(['perdicera', 'Perdicera', 'Águila perdicera', 'Aguila perdicera'])
index_real = df['especie'].isin(['real', 'Real', 'Águila real', 'Aguila real'])
index_buitre_negro = df['especie'].isin(['buitre negro', 'Buitre negro'])

# df.loc[index_imperial,'especie'] = 'aguila imperial'
# df.loc[index_perdicera,'especie'] = 'aguila perdicera'
# df.loc[index_real,'especie'] = 'aguila real'
# df.loc[index_buitre_negro,'especie'] = 'buitre negro'
# %% PROCESADO
variables =  ['peso', 'izda L', 'izda DV', 'dcha L', 'dcha DV', 'antebrazo', 'clave']
condition = (df['especie']=='buitre negro') & (df['edad'] == 'pollo') & (df['sexo'].isin(['macho', 'hembra']))
X_df = df.loc[condition,variables].apply(pd.to_numeric) 
X_df['L'] = X_df.loc[:,['izda L', 'dcha L']].mean(axis=1)
X_df['DV'] = X_df.loc[:,['izda DV', 'dcha DV']].mean(axis=1)
calculated_variables =  ['peso','L', 'DV', 'antebrazo','clave']
X_df = X_df.loc[:,calculated_variables]
index_notnull = ~pd.isnull(X_df).any(1)
condition = condition & index_notnull
    
X_df = X_df.dropna()
Y_df = df.loc[condition, 'sexo']
Y_df.replace(['macho', 'hembra'],[0, 1], inplace=True)

# %% Análisis
def normalize (X, var_mean = None, var_std = None):
    if var_mean is None and var_std is None:
        var_mean = X.mean(axis=0)
        var_std = X.std(axis=0)
    
        X_norm = (X-var_mean)/var_std
        return X_norm, var_mean, var_std
    else:
        X_norm = (X-var_mean)/var_std
        return X_norm
    
def Prediction_model (X_df, Y_df, n_splits, classifier):
    X = X_df.to_numpy()
    Y = Y_df.to_numpy()
    
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X)
    
    accuracy = np.zeros((n_splits,))
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        X_train_norm, var_mean, var_std = normalize(X_train)
        X_test_norm = normalize(X_test, var_mean=var_mean, var_std=var_std)
        
        model = classifier.fit(X_train_norm, Y_train)
        Y_predict = model.predict(X_test_norm)
        accuracy[i] = accuracy_score(Y_test, Y_predict)
        
    print('Media: ', accuracy.mean())
    print('Std: ', accuracy.std())
    
Prediction_model(X_df, Y_df, 10, LogisticRegression())
Prediction_model(X_df, Y_df, 10, LinearDiscriminantAnalysis())
Prediction_model(X_df, Y_df, 10, RandomForestClassifier(max_depth=2))
Prediction_model(X_df, Y_df, 10, SVC())


    
    
    



