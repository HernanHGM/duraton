# %% FILE DESCRIPTION
"""
Created on Sun Oct 29 20:51:21 2023

@author: HernanHGM

Fit variables distributions to the combination of two gaussian distributions 
associated to each sex

NO FUNCIONA, FALTAN DATOS PARA PODER HACER UN BUEN AJUSTE POR VARIABLE
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
df = df[df['Especie']=='Perdicera']
# %% SELECCIONO COLUMNAS
# Filtra las columnas de tipo float
float_columns = df\
        .select_dtypes(include=['float'])\
        .drop(labels='carpo', axis=1)\
        .columns\
        .tolist()
# %% PLOT
plt.close('all')
for col in float_columns:
    df.hist(col, bins=10)
# %% GAUSSIAN FIT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from scipy.optimize import curve_fit
from scipy.stats import betabinom, norm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, r2_score
from sklearn.preprocessing import MinMaxScaler

def data_fit(df, column, plot=False):
    data, bin_centers, variable = _get_fit_data(df, column)
    print(len(data), len(bin_centers))
    fit_coefs, cov_matrix, R2 = _optimize_parameters(bin_centers, 
                                                 data,
                                                 variable, 
                                                 plot=plot)
    return R2 #fit_coefs, cov_matrix

def _optimize_parameters(x, freq, variable, plot=False):
    '''
    Ajusta los 

    Parameters
    ----------
    x : np.array
        eje x de los datos a ajuste.
    freq : np.array
        eje y de los datos a ajusta, representa las frecuencias normalizadas del histograma.
    variable : str
        nombre variable
    plot : bool, optional
        True--> plot real and fitted data.
        False--> do not fit. 
        The default is False.

    Returns
    -------
    fit_coefs : np.array shape(6,)
        fitting parameters optimized.
    cov_matrix : np.matriz
        fit_coefs coveriance matrix.
        read scipy documentation:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    '''

    mu1 = min(x) + (max(x) - min(x))/4
    mu2 = min(x) + 3*(max(x) - min(x))/4
    sigma = np.std(x)
    A = 0.5
    p0 = [A, mu1, sigma, A, mu2, sigma]
    try:
        fit_coefs, cov_matrix = curve_fit(_fit_gauss, x, freq, p0=p0)
    except:
        fit_coefs, cov_matrix = p0, np.zeros((0,0))
        
    if plot==True:
        fig, ax = plt.subplots()
        ax.plot(x, freq, label='True data')
        y_pred = _fit_gauss(x, *fit_coefs)
        R2 = r2_score(freq, y_pred)
        ax.plot(x,
                y_pred,
                 marker='o', linestyle='',
                 label='Fit result')
        ax.set_xlabel('Bird speed (km/h)')
        ax.set_ylabel('Frequency normalized')            
        # specie = self.df.specie.unique()[0]
        # name = self.df.name.unique()[0]
        ax.set_title(f'r2_score: {R2}, variable: {variable}')
        
    return fit_coefs, cov_matrix, R2

def _get_fit_data(df, column):
    data = np.array(df[column])
    freq, bin_edges = np.histogram(data, 12, density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return freq, bin_centers, column

def _fit_gauss(x, A1, mu1, sigma1, A2, mu2, sigma2):
    """
    Función que describe un modelo de suma de dos gaussianas.

    Parámetros:
    -----------
    x : array-like
        Valores de entrada en el eje x.

    A1, mu1, sigma1, A2, mu2, sigma2 : float
        Parámetros del modelo.

    Retorna:
    --------
    y : array-like
        Valores calculados por el modelo para los valores de entrada x.
    """
    # Calcula la suma de dos gaussianas utilizando los parámetros proporcionados.
    gauss1 = A1 * norm.pdf(x, loc=mu1, scale=sigma1)
    gauss2 = A2 * norm.pdf(x, loc=mu2, scale=sigma2)
    y = gauss1 + gauss2

    return y
# %%
plt.close('all')
results = [(data_fit(df, col, plot=True), col) for col in float_columns]
