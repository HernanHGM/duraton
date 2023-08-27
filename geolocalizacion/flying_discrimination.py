import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from scipy.optimize import curve_fit
from scipy.stats import betabinom

from geolocalizacion import utils
# =============================================================================
# FLY ANALYSIS
# =============================================================================


class FlightAnalyzer:
    '''
    Clase que contiene las funciones necesarias para calcular cuando 
    un pájaro está en vuelo o posado
    
    get_histogram: calcula las frecuencias del histograma para tantos bins
    como valores enteros tiene la variable a calcular. 
    
    fit_function: Ajusta 
    
    '''
    
    def __init__(self):
        pass
    

    def get_histogram(self, df, column='speed_km_h', plot=False):
        '''
        Recibe un dataframe y extrae la columna indicada, por defecto se calcula la velocidad
        Devuelve los valores del eje x, su primer y ultimo valor y las frequencias
        '''
        self.df = df
        data = np.array(df[column])
        n_start = min(data)
        n_end = max(data)
        n_bins = n_end - n_start
    
        freq, x = np.histogram(data, n_bins, density=True)
        
        if plot==True:
            fig, ax = plt.subplots()
            df.hist(column, bins=len(x), log=True, ax = ax)
            ax.set_xlabel('Bird speed (km/h)')
            ax.set_ylabel('Frequency') 
            specie = df.especie.unique()[0]
            name = df.nombre.unique()[0]
            ax.set_title(f'Especie:{specie}, Nombre:{name}')
            
    
        return x[n_start:n_end], freq[n_start:n_end], n_start, n_end


    def fit_betabinom(self, x, C1, a1, b1, C2, a2, b2, n):
        '''
        Sum of two betabinomial function
        x: value in the axis x
        C1: normalizing constant for the first distribution
        a1: alfa value for the first betabinomial distribution
        b1: beta value for the first betabinomial distribution
        C2: normalizing constant for the second distribution
        a2: alfa value for the second betabinomial distribution
        b2: beta value for the second betabinomial distribution
        n: number of values that has the array to fit
        '''
        p1 = C1*betabinom.pmf(x, n, a1, b1)
        p2 = C2*betabinom.pmf(x, n, a2, b2)
        return p1+p2
    
    
    
    def optimize_parameters(self, 
                            x, freq, p0=[0.80, 1.5, 1000, 0.2, 5, 15],
                            plot=False):
        '''
        Ajusta los 

        Parameters
        ----------
        x : np.array
            eje x de los datos a ajuste.
        freq : np.array
            eje y de los datos a ajusta, representa las frecuencias normalizadas del histograma.
        p0 : list, optional
            parámetros iniciales de ajuste.
            The default is [0.80, 1.5, 1000, 0.2, 5, 15].
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
        n = len(freq)
        fit_function_partial = partial(self.fit_betabinom, n=n)
        fit_coefs, cov_matrix = curve_fit(fit_function_partial, x, freq, p0=p0)
        
        if plot==True:
            fig, ax = plt.subplots()
            ax.plot(x, freq, label='True data')
    
            ax.plot(x,
                     self.fit_betabinom(x, *fit_coefs, n=len(freq)),
                     marker='o', linestyle='',
                     label='Fit result')
            ax.set_xlabel('Bird speed (km/h)')
            ax.set_ylabel('Frequency normalized')            
            specie = self.df.especie.unique()[0]
            name = self.df.nombre.unique()[0]
            ax.set_title(f'Especie:{specie}, Nombre:{name}')
            
        return fit_coefs, cov_matrix
    
    def betabinomial(self, x, C1, a1, b1):
        '''
        beta binomial function

        Parameters
        ----------
        x : np.array
            x axis.
        C1 : float
            normalizacion constant.
        a1 : float
            alfa constant binomial distribution.
        b1 : float
            beta alfa constant binomial distribution.

        Returns
        -------
        p1 : np.array
            probability distribution along x array.

        '''
        n = len(x)
        p1 = C1*betabinom.pmf(x, n, a1, b1)
        return p1
    
    def find_flying_uncertainty(self, 
                                 x, freq, fit_coefs, 
                                 threshold=0.2, 
                                 n=3, 
                                 operation='>', 
                                 plot=False,
                                 scaling=True):
        '''
        finds values that are located in the uncertainty boundary where is not 
        clear if is flyng or not.
        
        Only the data ander the percentil 90% is analised because there it is
        the range where the uncertain values are located. For higher perncentiles
        the fitting error raises but it is clear than higher speeds than 40kmh
        belongs to a bird flying.
    
        Parameters
        ----------
        x : np.array
            x axis (meant to be speed_km_h, but could be any other magnitude)
        freq : np.array
            y axis (frequenciy normalized of the speed distribution).
        fit_coefs : np.array shape(6,)
            parameters of the curve fitting to two betabinomial distributions.
        threshold : float, optional
            Threshold that defines what values clearly belongs to one of the 
            distributions or they are doutful.
            The default is 0.2.
        n : int, optional
            number of consecutive values that must be in the uncertainty area 
            to be considered as uncertainty. 
            The default is 3.
        operation : str, optional
            operation to determine uncertain values.
            The default is '>'.
        plot : bool, optional
            True--> plot real, fitted data and uncertainty values.
            False--> do not fit. 
            The default is False.
        scaling : bool, optional (to set the uncertain values to the same scale)
            True--> scale uncertainty values.
            False--> do not scale. 
            The default is True.
    
        Returns
        -------
        uncertain_values : list
            list of consecutive index whose values meet the predefined condition.
    
        '''
        freq_cum= np.cumsum(freq)
        # Fijamos el limite en percentil 90 porque el ~80% de los datos 
        # corresponden a datos de bajas velocidades <10kmh
        # y el ~20% restante a datos de altas velocidades >10kmh
        # Como solo queremos analizar el tramo de ~10kmh tomamos solo el 90% de
        # los datos para no considerar la cola final de altas velocidades.
        # Evitamos esta cola final porque el error de ajuste aumenta, 
        # pero no hay duda de que son datos de vuelo y no posados
        limit_uncertainty = np.argmax(freq_cum>0.9)
        y1 = self.betabinomial(x, *fit_coefs[:3])
        y2 = self.betabinomial(x, *fit_coefs[3:])
        y3 = abs(freq-(y1+y2))/(y1+y2+freq)
        
        uncertain_values = utils.conditional_selection(y3[:limit_uncertainty], threshold, n, operation)
        if plot==True:
            plt.plot(x[:limit_uncertainty], y1[:limit_uncertainty], 'r.', label='Beta Binomial1')
            plt.plot(x[:limit_uncertainty], y2[:limit_uncertainty], 'b.', label='Beta Binomial1')
            if scaling==True:
                scale = round(max(y3[:limit_uncertainty])/max(y2), -2)
                print(f'To improve data visualization |true-pred|/(true+pred) has been scaled dividing by {scale}')
                print(f'uncertainty threshold = {threshold}')
                y3 = y3/scale
            
            plt.plot(x[uncertain_values], y3[uncertain_values], 'ko', label='uncertain values')
            plt.plot(x[:limit_uncertainty], y3[:limit_uncertainty], 'g.', label='|true-pred|/(true+pred)')
            plt.legend()
            
        return uncertain_values  
    

    def define_flying_situation(self, 
                                df: pd.DataFrame,
                                uncertain_values: list):
        """
        Define the flying situation categories based on speed values.
        There is a range of speeds where it is not clear if the bird is flying 
        or not.
        Under the minimun value of that range we know it is stopped
        Over the maximun value of that range we know it is flying
        Inside that range we dont know
        Parameters:
            df (pandas.DataFrame): The input DataFrame with flight data.
            uncertain_values (list): List of uncertain speed values.

        Returns:
            pandas.DataFrame: DataFrame with added 'flying_situation' column.
        """
        max_uncertain_speed = int(max(uncertain_values))
        min_uncertain_speed = int(min(uncertain_values))
        
        # Create a partial function with fixed parameters
        partial_assign = partial(self._assign_flying_categories, 
                                 max_uncertain_speed=max_uncertain_speed, 
                                 min_uncertain_speed=min_uncertain_speed)

        # Apply the partial function to each row of the DataFrame
        df['flying_situation'] = df.apply(partial_assign, axis=1)
        df['numerical_flying_situation'] = df.apply(self._assign_numerical_flying_categories, axis=1)
        
        return df
    
    def _assign_flying_categories(self, 
                                  row: pd.Series, 
                                  max_uncertain_speed: int, 
                                  min_uncertain_speed: int):
        """
        Assign flying categories based on speed values.
        There is a range of speeds where it is not clear if the bird is flying 
        or not.
        Under the minimun value of that range we know it is stopped
        Over the maximun value of that range we know it is flying
        Inside that range we dont know
        Parameters:
            row (pandas.Series): A row from the DataFrame.
            max_uncertain_speed (int): Maximum uncertain speed value.
            min_uncertain_speed (int): Minimum uncertain speed value.

        Returns:
            str: Flying category ('flying', 'landed', 'undefined').
        """
        if row['speed_km_h'] > max_uncertain_speed:
            return 'flying'
        elif row['speed_km_h'] < min_uncertain_speed:
            return 'landed'
        else:
            return 'undefined'
        
    def _assign_numerical_flying_categories(self,
                                  row: pd.Series):
        """
        Assign a numerical value 0, or 1 based on the nominal value
        of the column flying_situation.
        """
        if row['flying_situation'] == 'landed':
            return 0
        elif row['flying_situation'] == 'flying':
            return 1
