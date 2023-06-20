# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:26:19 2023

@author: Hernán García Mayoral
"""
import numpy as np
import pandas as pd
from statistics import mode
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class SelectorClassifier:
    
    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.features = [x for x in self.df.columns if x != self.label]
        
        columns_by_model = {'mean_kappa': [],
                            'std_kappa': [], 
                            'mean_accuracy': [],
                            'std_accuracy': [],
                            'used_features': [],
                            'classifier_model': []}
        self.results_by_model = pd.DataFrame(columns_by_model) 

        columns_by_features = {'n_features': [],
                               'mean_kappa': [],
                               'std_kappa': [], 
                               'mean_accuracy': [],
                               'std_accuracy': [],
                               'classifier': []}
        self.results_by_features = pd.DataFrame(columns_by_features) 

    def select_best_classifier(self, classifier_dict, n_splits = 10):
        '''
        Busca el mejor classificador entre los datos, además recoge
        todos los features que aportan información en la clasificación.
        Divide los datos en n_splits e itera buscando para cada combinación
        de datos los mejores features para el clasificador dado.
        Los mejores features se calculan con la funcion SelectFromModel
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
        Para conocer la eficacia del clasificador se calculan kappa y accuracy.
        Parameters
        ----------
        classifier : modelo de scikit learn
            modelo de clasificacion.
        n_splits : int, optional
            numero de splits en los que se particionan los datos. 
            The default is 10.
    
        Returns
        -------
        results : pd.DataFrame
            mean and std of kappa and accuray and the used features by the classifier.

    
        '''
        classifier_model = list(classifier_dict.values())[0]
        selector = self._choose_selector(classifier_model)
        X, Y, features = self.feature_label_separator()
        
        kf = StratifiedKFold(n_splits = n_splits, 
                             shuffle = True, 
                             random_state = 42)
        kappa = []
        accuracy = []
        selected_features = []
        for train_index, test_index in kf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            selector.fit(X_train, Y_train)
            X_train_reduced = selector.transform(X_train)
            X_test_reduced = selector.transform(X_test)
            selected_features.append(tuple(features[selector.get_support()]))
            
            classifier_model.fit(X_train_reduced, Y_train)
            Y_pred = classifier_model.predict(X_test_reduced)
            Y_pred_round = np.round(Y_pred)
            kappa.append(cohen_kappa_score(Y_test, Y_pred_round))
            accuracy.append(accuracy_score(Y_test, Y_pred_round))
        
        self._summarize_results_by_model(classifier_dict, 
                                         selected_features, 
                                         kappa, 
                                         accuracy)  
        
    def select_best_features(self, n_splits = 10):
        '''
        Se usa el clasificador que obtuvo la mejor kappa en 
        self.select_best_classifier. junto a los features usados para hallar la 
        mejor combinación de esos features.

        Parameters
        ----------
        n_splits : int, optional
            número de particiones del dataset. The default is 10.

        Returns
        -------
        None
        '''
        classifier_name, classifier, used_features = self._get_best_model()
        X, Y, features = self.feature_label_separator(valid_columns=used_features)
        
        n_features = len(features)
       
        kf = StratifiedKFold(n_splits = n_splits,
                             shuffle = True,
                             random_state = 42)
                
        start = time.process_time()
        for i in range(1, n_features+1):
            if i<n_features:
                sfs = SequentialFeatureSelector(classifier, 
                                                n_features_to_select = i)
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
                    
                    classifier.fit(X_train_reduced, Y_train)
                    Y_pred = classifier.predict(X_test_reduced)
                    Y_pred_round = np.round(Y_pred)
                    
                    kappa.append(cohen_kappa_score(Y_test, Y_pred_round))
                    accuracy.append(accuracy_score(Y_test, Y_pred_round))
                    
            if i == n_features:
                kappa = []
                accuracy = []
                selected_features = []
                for train_index, test_index in kf.split(X, Y):
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]
                    
                    selected_features.append(tuple(features))
                    
                    classifier.fit(X_train, Y_train)
                    Y_pred = classifier.predict(X_test)
                    Y_pred_round = np.round(Y_pred)
                    
                    kappa.append(cohen_kappa_score(Y_test, Y_pred_round))
                    accuracy.append(accuracy_score(Y_test, Y_pred_round))
                
            
            print('n features used = ', i)
            print('Tiempo transcurrido = ', time.process_time() - start, ' segundos')
            
            features_list = mode(selected_features) 
            self._summarize_results_by_features(classifier_name, 
                                                features_list, 
                                                kappa,
                                                accuracy)
        
    def feature_label_separator(self, valid_columns=[]):
        if valid_columns==[]:
            valid_columns = self.features
        else:      
            valid_columns = [x for x in valid_columns if x != self.label]
            
        X = self.df[valid_columns].values
        Y = np.squeeze(self.df[self.label].values)
        return X, Y, np.array(valid_columns)  
    ###########################################################################
    # PRIVATE METHODS
    ###########################################################################
    def _convert_numeric(self, df):
        numeric_cols = ['mean_kappa','std_kappa','mean_accuracy','std_accuracy']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        return df
    
        
    def _check_classifier(self, classifier_model):
        cond1 = hasattr(classifier_model, 'coef_')
        cond2 = hasattr(classifier_model, 'feature_importances_')
        valid_selector = any((cond1, cond2))
        return valid_selector
        
    def _choose_selector(self, classifier_model):
        '''
        Detects if classifier is valid to be used by SelectFromModel.
        In case it is not LDA is used

        Parameters
        ----------
        classifier_model : scikit learn classifier
            Model used for classification.

        Returns
        -------
        selector : scikit learn selector
            model used for feature selection.
        '''
        valid_selector = self._check_classifier(classifier_model)
        if valid_selector:
            selector = SelectFromModel(estimator=classifier_model)
        else:
            selector = SelectFromModel(estimator=LinearDiscriminantAnalysis())
        return selector
        

    
    def _get_best_model(self):
        # Encontrar el indice con mayor valor de kappa
        best_classifier_name = self.results_by_model['mean_kappa'].idxmax()
        # Obtener el clasificador con mejor kappa
        best_classifier_model = self.results_by_model.loc[best_classifier_name , 'classifier_model']
        # Obtener los features usados por el clasificador con mejor kappa
        used_features = self.results_by_model.loc[best_classifier_name , 'used_features']
        return best_classifier_name, best_classifier_model, used_features
    
    
    
    def _summarize_results_by_model(self, 
                                    classifier_dict, 
                                    selected_features, 
                                    kappa, 
                                    accuracy):
        '''
        Creates a pandas datafrfame summarizing the metrics of the classifier

        Parameters
        ----------
        classifier_dict : dict
            key: name of the classifier
            value: classifier itself used for feature selection and classification
        selected_features : list of tuples of strings
            list containing the tuples of the features selected in each k-fold.
        kappa : list of floats
            list containing the kappa scores for each k-fold.
        accuracy : list of floats
            list containing the kappa scores for each k-fold.

        Returns
        -------
        results : pd.DataFrame
            mean and std of kappa and accuray and the used features by the classifier.

        '''
        all_features = [string for tupla in selected_features for string in tupla]
        used_features = list(set(all_features))
        mean_kappa = np.mean(kappa).round(2)
        std_kappa =  np.std(kappa).round(2)
        mean_accuracy = np.mean(accuracy).round(2)
        std_accuracy =  np.std(accuracy).round(2)
        
        classifier_name = list(classifier_dict.keys())[0]
        classifier_model = list(classifier_dict.values())[0]
        
        data = np.array([mean_kappa, std_kappa, 
                         mean_accuracy, std_accuracy,
                         used_features, classifier_model], dtype=object)
       
        self.results_by_model.loc[classifier_name] = data
        self.results_by_model = self._convert_numeric(self.results_by_model)
           
    

    def _summarize_results_by_features(self, 
                                       classifier_name, 
                                       features_list, 
                                       kappa,
                                       accuracy):
        '''
        Creates a pandas datafrfame summarizing the metrics of the used features

        Parameters
        ----------
        classifier_name : str
            classifier used for feature selection and classification.
        features_list : list of strings
            list containing the most features used.
        kappa : list of floats
            list containing the kappa scores for each k-fold.
        accuracy : list of floats
            list containing the kappa scores for each k-fold.

        Returns
        -------
        results : pd.DataFrame
            mean and std of kappa and accuray and the classifier by used features.

        '''
        
        n_features = len(features_list)
        features_str = '//'.join(features_list)
        mean_kappa = np.mean(kappa).round(2)
        std_kappa =  np.std(kappa).round(2)
        mean_accuracy = np.mean(accuracy).round(2)
        std_accuracy =  np.std(accuracy).round(2)
        
        data = np.array([n_features,
                         mean_kappa, std_kappa, 
                         mean_accuracy, std_accuracy,
                         classifier_name], dtype=object)
        
        self.results_by_features.loc[features_str] = data
        self.results_by_features = self._convert_numeric(self.results_by_features)

        

         
# print simbolo +-   u"\u00B1"