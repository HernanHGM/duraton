# %% IMPORT LIBRERIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

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
# %% LIMPIEZA Y PREPROCESADO

def individual_selection (df, especie, edad, label, variables):
    condition_especie = (df['especie']==especie)
    condition_edad = (df['edad'] == edad) 
    condition_sexo = (df['sexo'].isin(['macho', 'hembra']))
    
    global_condition = condition_especie & condition_edad & condition_sexo
    
    df_var = df.loc[global_condition, variables].apply(pd.to_numeric) 
    df_label = df.loc[global_condition, label]
    
    df_selected = pd.concat((df_label, df_var), axis = 1)
    
    return df_selected

def scaling_encoding(df, label, *variables):
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    # data.columns = data.columns.str.title()
    
    if variables:
        predictors = df[variables[0]]
        X = pd.DataFrame(MinMaxScaler().fit_transform(predictors), columns = predictors.columns)
    else:
        predictors = df.drop(columns = label) 
        X = pd.DataFrame(MinMaxScaler().fit_transform(predictors), columns = predictors.columns)
    
    Y = pd.DataFrame(LabelEncoder().fit_transform(df[label]), columns = label)
    data_scaled = pd.concat((Y,X), axis = 1)
    
    return data_scaled, X, Y

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
    
    return df_clean
 

   
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


data = individual_selection(df_original, 'Buitre negro', 'pollo', label, variables)
data_scaled, _, _ = scaling_encoding(data, label, variables)
data_scaled2 = remove_outliers(data_scaled, label)
data_scaled3 = drop_nan_columns(data_scaled2, 0.1)

# %% VISUAL ANALYSIS
def features_distribution(data):
    warnings.filterwarnings("ignore") # ignore it is a deprecated function and there is also a new one better

    plt.figure(figsize = [20, 10])
    cols = data.columns
    contador = 1
    for col in cols:
        plt.subplot(5, 3, contador)
        sns.distplot(data[col], hist_kws = dict(edgecolor = 'k', linewidth = 1, color = 'crimson'), color = 'red')
        contador += 1

    plt.tight_layout()
    return plt.show()

features_distribution(data_def)
# %% FEATURE EXTRACTION

data_scaled3['L DV izda'] = data_scaled3['izda L'] * data_scaled3['izda DV'] 
data_scaled3['L DV dcha'] = data_scaled3['dcha L'] * data_scaled3['dcha DV'] 

data_scaled3['L mean'] = data_scaled3.loc[:,['izda L', 'dcha L']].mean(axis=1)
data_scaled3['DV mean'] = data_scaled3.loc[:,['izda DV', 'dcha DV']].mean(axis=1)
data_scaled3['L DV mean'] = data_scaled3.loc[:,['L DV izda', 'L DV dcha']].mean(axis=1)
data_scaled3['L DV mean2'] = data_scaled3['L mean'] * data_scaled3['DV mean'] 

# data_scaled3['L max'] = data_scaled3.loc[:,['izda L', 'dcha L']].max(axis=1)
# data_scaled3['DV max'] = data_scaled3.loc[:,['izda DV', 'dcha DV']].max(axis=1)
# data_scaled3['L DV max'] = data_scaled3.loc[:,['L DV izda', 'L DV dcha']].max(axis=1)

# data_scaled3['L min'] = data_scaled3.loc[:,['izda L', 'dcha L']].min(axis=1)
# data_scaled3['DV min'] = data_scaled3.loc[:,['izda DV', 'dcha DV']].min(axis=1)
# data_scaled3['L DV min'] = data_scaled3.loc[:,['L DV izda', 'L DV dcha']].min(axis=1)

data_new = data_scaled3.drop(columns = ['izda L', 'dcha L', 'izda DV', 'dcha DV', 'L DV dcha', 'L DV izda'])
data_new, _, _ = scaling_encoding(data_new, label)
data_new2 = data_new.drop(columns = ['ala d', 'ala v', '7º  1ª', 'cola', 'rectrix c', 'cañón en 7ª', 'antebrazo'])
def matrix_and_relevant_relationships(df, threshold):
    # gradient = np.linspace(-1, 1)
    # gradient = np.vstack((gradient, gradient))
    # my_cmap = sns.diverging_palette(9, 9, s = 100, l = 33, sep = 30, n = 100, center = 'light', as_cmap = True)
    # fig, ax = plt.subplots(1, figsize=(20, 1))
    # ax.imshow(gradient, aspect='auto', cmap = my_cmap)
    # ax.set_axis_off()
    # plt.show()

    correlation_matrix = df.corr(method = 'pearson')
    # display(correlation_matrix.style.background_gradient(axis = None, cmap = my_cmap))
    
    condition = (abs(correlation_matrix) > threshold) & (correlation_matrix != 1)
    upper_matrix_condition = np.triu((np.ones(correlation_matrix.shape)).astype(bool))

    relationships = correlation_matrix[upper_matrix_condition & condition]
    relationships = relationships.stack().reset_index()
    relationships.columns = ['feature_1', 'feature_2', 'correlation']
    relationships = relationships.reindex(relationships.correlation.abs().sort_values(ascending = False).index).reset_index(drop = True)
    return correlation_matrix, relationships

def creating_interacting_features(data, relations):
   
    df_new_features = pd.DataFrame()
    for i in range(len(relations)):
        feature_name1 = relations.loc[i,'feature_1']
        feature_name2 = relations.loc[i,'feature_2']
        feature_1 = data[feature_name1]
        feature_2 = data[feature_name2]
       
        new_feature = feature_1*feature_2
       
        df_new_features[feature_name1+' '+feature_name2] = new_feature
    augmented_data = pd.concat((data, df_new_features), axis = 1)
    return augmented_data, df_new_features

variables_sel = ['sexo', 'peso', 'izda L', 'izda DV', 'dcha L', 'dcha DV', 'antebrazo', 'cañón en 7ª']
correlation_matrix1, relations = matrix_and_relevant_relationships(data_new2, 0.5)
augmented_data, df_new_features = creating_interacting_features(data_new2, relations)

cm = augmented_data.corr()

# %%
# =============================================================================
# Classifiers
# =============================================================================
RF_C = RandomForestClassifier(n_estimators = 100, max_depth = 2, min_samples_split=10, random_state = 0)
LogReg = LogisticRegression()
SVM = SVC(class_weight='balanced')
LDA = LinearDiscriminantAnalysis()
KNN_C = KNeighborsClassifier(n_neighbors = 5)

# =============================================================================
# Regressors
# =============================================================================
GBR = GradientBoostingRegressor(n_estimators = 100, random_state = 0)
LinReg = LinearRegression()
KNN_R = KNeighborsRegressor(n_neighbors = 5)
RF_R = RandomForestRegressor(n_estimators = 100, random_state = 0)

# =============================================================================
# Feature Selectors: Extractors
# =============================================================================
pca = PCA(n_components = 2)
lda = LinearDiscriminantAnalysis()

# =============================================================================
# Feature Selectors: Wrapped methods
# =============================================================================
sel_model = SelectFromModel(RF_C)

def selector_and_predictor(X, Y, model, model_selector, n_splits = 10):
    import time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
    
    start = time.process_time()
    classes = np.unique(Y)
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
        cm.append(confusion_matrix(Y_test, Y_pred_round, labels = classes))
    cm = np.array(cm)
    print('Tiempo transcurrido = ', time.process_time() - start, ' segundos')
    print('kappa = ', np.mean(kappa).round(2), u"\u00B1", np.std(kappa).round(2))
    print('accuracy = ', np.mean(accuracy).round(2), u"\u00B1", np.std(accuracy).round(2))
    # disp = ConfusionMatrixDisplay(confusion_matrix = np.mean(cm, axis=0).round(2), display_labels = classes)
    # disp.plot(cmap = 'Reds')
    # plt.show()
    
data_def = data_new2.drop(columns = ['L DV mean', 'peso']).dropna()    
X = data_def.iloc[:,1:].to_numpy()#drop(columns = 'sexo').to_numpy()
Y = data_def['sexo'].to_numpy()
selector_and_predictor(X, Y , SVM, pca, n_splits = 10)

# %% EXTRA
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
        
        # X_train_norm, var_mean, var_std = normalize(X_train)
        # X_test_norm = normalize(X_test, var_mean=var_mean, var_std=var_std)
        
        model = classifier.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
        accuracy[i] = accuracy_score(Y_test, Y_predict)
        
    print('Media: ', accuracy.mean())
    print('Std: ', accuracy.std())
    
# Prediction_model(X_df, Y_df, 10, LogisticRegression())
# Prediction_model(X_df, Y_df, 10, LinearDiscriminantAnalysis())
# Prediction_model(X_df, Y_df, 10, RandomForestClassifier(max_depth=2))
# Prediction_model(X_df, Y_df, 10, SVC())