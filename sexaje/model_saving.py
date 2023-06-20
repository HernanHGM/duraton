# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:23:28 2023

@author: Usuario
"""

import numpy as np
from joblib import dump, load
from sklearn.metrics import cohen_kappa_score


import sys
path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)
from sexaje.model_creation import SelectorClassifier
from sexaje import parameters

path = 'E:\\duraton\\sexaje\\_dev\\Buitre Negro\\model_scaler'

    
def get_best_features(features_results):
    max_kappa = features_results.mean_kappa.max()
    
    best_features_list = list(features_results[features_results.mean_kappa == max_kappa].index)
    best_features_list = [features.split('//') for features in best_features_list]
    features_list = best_features_list[0]
    
    classifier = list(features_results['classifier'])
    classifier = classifier[0]
    
    accuracy = list(features_results[features_results.mean_kappa == max_kappa].mean_accuracy)
    accuracy = accuracy[0]

    return features_list, classifier, max_kappa, accuracy

def check_scaler(X, scaler_filename, X_scaled):
    scaler = load(scaler_filename)
    X_scaled2 = scaler.transform(X)
    
    if (X_scaled==X_scaled2).all():
        print('scaler ok')
    else:
        raise ValueError('X2 != X3')
        
def check_classifier(X_scaled, Y, classifier_filename, kappa_baseline):
    classifier = load(classifier_filename)
    
    Y_pred = classifier.predict(X_scaled)
    Y_pred_round = np.round(Y_pred)
    
    new_kappa = np.round(cohen_kappa_score(Y, Y_pred_round),2)
    if (new_kappa>=kappa_baseline):
        print('classifier ok')
    else:
        print(f'new kappa = {new_kappa}')
        print(f'kappa baseline = {kappa_baseline}')
        raise ValueError('new_kappa < kappa_baseline')
        
def write_documentation(saving_path, conteos_str, 
                        classifier, scaler,
                        features_list, max_kappa, accuracy):
    filename = '\\'.join((saving_path, 'Model_specifications.txt'))
    features_txt = '  '.join(features_list)
    with open(filename, 'a') as f:
        f.write(conteos_str)
        f.write('\n')
        f.write(f'features_list: {features_txt}')
        f.write('\n')
        f.write(f'Scaler: {scaler}')
        f.write('\n')
        f.write(f'Classifier: {classifier}')
        f.write('\n')
        f.write('\n')
        f.write('Values obtained during trainning with k-fold=10:')
        f.write('\n')
        f.write(f'kappa={max_kappa}; accuracy={accuracy}')
        f.write('\n')
        

    
def save_models(df, label, conteos_str, features_results, 
                saving_path, scaler_name):
    
    features_list, classifier_name, max_kappa, accuracy = get_best_features(features_results)
    
    scaler = parameters.scaler_dict[scaler_name]
    SC = SelectorClassifier(df, label)
    X,Y,_ = SC.feature_label_separator(features_list)
    X_scaled = scaler.fit_transform(X)
   
    scaler_filename = '\\'.join((saving_path, 'scaler.joblib'))
    dump(scaler, scaler_filename)
    check_scaler(X, scaler_filename, X_scaled)
    
    classifier = parameters.classifier_dict[classifier_name]
    classifier.fit(X_scaled, Y)
    classifier_filename = '\\'.join((saving_path, 'classifier.joblib'))
    dump(classifier, classifier_filename)
    
    check_classifier(X_scaled, Y, classifier_filename, max_kappa)
    
    write_documentation(saving_path, conteos_str, 
                        classifier_name, scaler_name,
                        features_list, max_kappa, accuracy)
    
