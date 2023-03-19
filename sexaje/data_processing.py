# =============================================================================
# Data Selection
# =============================================================================
def individual_selection (df, especie, edad, label, variables):
    import pandas as pd
    
    condition_especie = (df['especie']==especie)
    condition_edad = (df['edad'].isin(edad)) 
    condition_sexo = (df['sexo'].isin(['macho', 'hembra']))
    
    global_condition = condition_especie & condition_edad & condition_sexo
    
    df_var = df.loc[global_condition, variables].apply(pd.to_numeric) 
    df_label = df.loc[global_condition, label]
    
    df_selected = pd.concat((df_label, df_var), axis = 1)
    
    return df_selected

# =============================================================================
# Data cleaning and preprocessing
# =============================================================================
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
    import pandas as pd
    
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
    # Drop columns with too many nans
    na_proportion = df.isna().sum()/len(df)
    condition = na_proportion <= threshold
    df_clean = df.loc[:,condition]
    
    # Drop the rows with any nans
    df_clean = df_clean.dropna()
    
    return df_clean

def scaling_encoding(df_original, label, *variables):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    # Si no creo esta copia, el dataframe que pase se verá afectado 
    # por las operaciones de esta función. Pq df es un objeto mutable
    df = df_original.copy()
    
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

def standarizer(df_original, label, *variables):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    # Si no creo esta copia, el dataframe que pase se verá afectado 
    # por las operaciones de esta función. Pq df es un objeto mutable
    df = df_original.copy()
    
    if variables:
        predictors = df[variables[0]]
        scaler = StandardScaler().fit(predictors)
        x_scaled = scaler.transform(predictors) + scaler.mean_
        X = pd.DataFrame(x_scaled, columns = predictors.columns)
        
    else:
        predictors = df.drop(columns = label) 
        scaler = StandardScaler().fit(predictors)
        x_scaled = scaler.transform(predictors) + scaler.mean_      
        X = pd.DataFrame(x_scaled, columns = predictors.columns)
        
    #np.squeeze because fit_transform needs shape (n,)
    Y = pd.DataFrame(LabelEncoder().fit_transform(np.squeeze(df[label])), columns = label)
    data_scaled = pd.concat((Y,X), axis = 1)
    
    return data_scaled, X, Y

# =============================================================================
# Model
# =============================================================================

def selector_and_predictor(X, Y, model, model_selector, n_splits = 10):
    import numpy as np
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
        
        model_selector.fit(X_train, Y_train)
        X_train_reduced = model_selector.transform(X_train)
        X_test_reduced = model_selector.transform(X_test)
        
        model.fit(X_train_reduced, Y_train)
        Y_pred = model.predict(X_test_reduced)
        Y_pred_round = np.round(Y_pred)
        kappa.append( cohen_kappa_score(Y_test, Y_pred_round))
        accuracy.append(accuracy_score(Y_test, Y_pred_round))
        cm.append(confusion_matrix(Y_test, Y_pred_round))
    cm_array = np.array(cm)   
    print('Tiempo transcurrido = ', time.process_time() - start, ' segundos')
    print('kappa = ', np.mean(kappa).round(2), u"\u00B1", np.std(kappa).round(2))
    print('accuracy = ', np.mean(accuracy).round(2), u"\u00B1", np.std(accuracy).round(2))

    
def best_features(X, Y, features, model, n_splits = 10, n_features = 5):
    import numpy as np
    import time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
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
        
        print('n features used = ', i)
        print('Tiempo transcurrido = ', time.process_time() - start, ' segundos')
        print('kappa = ', np.mean(kappa).round(2), u"\u00B1", np.std(kappa).round(2))
        print('accuracy = ', np.mean(accuracy).round(2), u"\u00B1", np.std(accuracy).round(2))
        print(mode(selected_features))
        print()
        
        most_selected_features.append(mode(selected_features))
        kappa_global.append([np.mean(kappa).round(2), np.std(kappa).round(2)])
        accuracy_global.append([np.mean(accuracy).round(2), np.std(accuracy).round(2)])
     

    return most_selected_features, kappa_global, accuracy_global

# =============================================================================
# MODEL CREATOR
# =============================================================================
def scaler_model_creator(data_clean, label, variable):   
    # Futura mejora: no necesitar el termino variable ni label
    import pickle
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    
    var_short = variable.replace(" ", "")
    
    # =============================================================================
    # SCALER
    # =============================================================================
    scaler = MinMaxScaler()
    
    x_tofit = data_clean[[variable]].values
    scaler.fit(x_tofit)
    
    filename = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\scaler_' + var_short + '.pkl'
    pickle.dump(scaler, open(filename, 'wb'))
    
    # =============================================================================
    # MODEL
    # =============================================================================
    data_def, X_scaled, Y_scaled = scaling_encoding(data_clean, label)
    X = X_scaled.values
    Y = np.squeeze(Y_scaled.values)
    
    model = LogisticRegression()
    model.fit(X, Y)
    
    filename = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\model_' + var_short + '.pkl'
    pickle.dump(model, open(filename, 'wb'))


# =============================================================================
# 
# =============================================================================
