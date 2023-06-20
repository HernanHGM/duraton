import numpy as np
# =============================================================================
# Data cleaning and preprocessing
# =============================================================================
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






# =============================================================================
# Model
# =============================================================================
def feature_label_separator(df, label, valid_columns=[]):
    if valid_columns==[]:
        valid_columns = df.columns
        
    features = [x for x in valid_columns if x != label]
    X = df[features].values
    Y = np.squeeze(df[label].values)
    return X, Y, np.array(features)
    
def selector_and_predictor(df, label, classifer, selector, n_splits = 10):
    '''
    Selecciona los mejores features dentro de X usando el modelo indicado
    en model_selector y crea un modelo de clasificacion usando el modelo dado
    por model para ajustar los datos de X a Y.
    Muestra el tiempo transcurrido para entrenar cada modelo, 
    el kappa medio, la precisión media y la matriz de confusión media de 
    las distintas particiones
    Parameters
    ----------
    X : array
        Los datos a clasificar
    Y : array
        Las etiquetas de clasificación
    classifer : modelo de scikit learn
        modelo de clasificacion.
    selector : modelo de scikit learn
        modelo de seleccion de features.
    n_splits : int, optional
        numero de splits en los que se particionan los datos. 
        The default is 10.

    Returns
    -------
    None.

    '''
    import numpy as np
    import time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
    
    X, Y, features = feature_label_separator(df, label)
    
    start = time.process_time()
    kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    kappa = []
    accuracy = []
    cm = []
    selected_features = []
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        selector.fit(X_train, Y_train)
        X_train_reduced = selector.transform(X_train)
        X_test_reduced = selector.transform(X_test)
        selected_features.append(tuple(features[selector.get_support()]))
        
        classifer.fit(X_train_reduced, Y_train)
        Y_pred = classifer.predict(X_test_reduced)
        Y_pred_round = np.round(Y_pred)
        kappa.append( cohen_kappa_score(Y_test, Y_pred_round))
        accuracy.append(accuracy_score(Y_test, Y_pred_round))
        cm.append(confusion_matrix(Y_test, Y_pred_round))
    cm_array = np.array(cm)   
    print('Tiempo transcurrido = ', time.process_time() - start, ' segundos')
    print('kappa = ', np.mean(kappa).round(2), \
                      u"\u00B1", \
                      np.std(kappa).round(2))
    print('accuracy = ', np.mean(accuracy).round(2), \
                         u"\u00B1", \
                         np.std(accuracy).round(2))
    print('confusion matix = ', np.mean(cm_array, axis = 0).round(2), \
                                u"\u00B1", \
                                np.std(cm_array, axis = 0).round(2))
    print(f'selected_features: {selected_features}')

    
def best_features(df, label, valid_columns, model, n_splits = 10, n_features = 5):
    import numpy as np
    import time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
    from statistics import mode
    
    X, Y, features = feature_label_separator(df, label, valid_columns)
    
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
def scaler_model_creator(df, label, feature, path):   
    # Futura mejora: no necesitar el termino variable ni label
    import pickle
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    
    # var_short = variable.replace(" ", "")
    X, Y, _ = feature_label_separator(df, label, feature)
    # =============================================================================
    # SCALER
    # =============================================================================
    scaler = MinMaxScaler()
    
    x_tofit = df[[feature]].values
    scaler.fit(x_tofit)
    
    filename = path + '\\scaler.pkl'
    pickle.dump(scaler, open(filename, 'wb'))
    
    # =============================================================================
    # MODEL
    # =============================================================================
    data_def, X_scaled, Y_scaled = scaling_encoding(data_clean, label)
    X = X_scaled.values
    Y = np.squeeze(Y_scaled.values)
    
    model = LogisticRegression()
    model.fit(X, Y)
    
    filename = path + '\\classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))


# =============================================================================
# 
# =============================================================================
