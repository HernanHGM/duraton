import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from scipy.optimize import curve_fit
from scipy.stats import betabinom

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler

from geolocalizacion import utils
# =============================================================================
# FLY ANALYSIS
# =============================================================================
# Notas mejora: asegurar que las gráficas de los datos de velocidades y su
# sean siempre la misma. Ahora salen bien pero es un tanto precario porque no
# se pasa el axis de la fiigura de datos puros al axis de la figura con el ajuste

class FlightAnalyzer:
    '''
    Class that contains the necessary functions to calculate whether a bird
    is flying or landed.
    
    get_histogram: Calculates the histogram frequencies for as many bins
    as there are integer values in the variable to be calculated.

    fit_function: Fits and

    
    '''
    
    def __init__(self, df):
        self.df = df
    

    def get_histogram(self, column='speed_km_h', plot=False):
        '''
        It receives a DataFrame and extracts the specified column; 
        by default, it calculates the speed. 
        It returns the values on the x-axis, their first and last values,
        and the frequencies.
        It can plot the histogram of the values
        '''
        data = np.round(np.array(self.df[column])).astype(int)
        n_start = min(data)
        n_end = max(data)
        n_bins = int(n_end - n_start)
    
        freq, x = np.histogram(data, n_bins, density=True)
        
        if plot==True:
            fig, ax = plt.subplots()
            self.df.hist(column, bins=len(x), log=True, ax = ax)
            ax.set_xlabel('Bird speed (km/h)')
            ax.set_ylabel('Frequency') 
            specie = self.df.specie.unique()[0]
            name = self.df.name.unique()[0]
            ax.set_title(f'Specie: {specie}, Name: {name}')
            
    
        return x[n_start:n_end], freq[n_start:n_end], n_start, n_end
    
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
        fit_function_partial = partial(self._fit_betabinom, n=n)
        fit_coefs, cov_matrix = curve_fit(fit_function_partial, x, freq, p0=p0)
        
        if plot==True:
            fig, ax = plt.subplots()
            ax.plot(x, freq, label='True data')
    
            ax.plot(x,
                     self._fit_betabinom(x, *fit_coefs, n=len(freq)),
                     marker='o', linestyle='',
                     label='Fit result')
            ax.set_xlabel('Bird speed (km/h)')
            ax.set_ylabel('Frequency normalized')            
            specie = self.df.specie.unique()[0]
            name = self.df.name.unique()[0]
            ax.set_title(f'Specie: {specie}, Name: {name}')
            
        return fit_coefs, cov_matrix
    
    def _fit_betabinom(self, x, C1, a1, b1, C2, a2, b2, n):
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
        
    def find_flying_uncertainty(self, 
                                 x, freq, fit_coefs, 
                                 threshold=0.4, 
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
        y1 = self._betabinomial(x, *fit_coefs[:3])
        y2 = self._betabinomial(x, *fit_coefs[3:])
        y3 = 2*abs(freq-(y1+y2))/(y1+y2+freq) #Diferencia/media multiplico *2 debido a la media
        
        uncertain_values = utils.conditional_selection(y3[:limit_uncertainty], threshold, n, operation)
        if plot==True:
            plt.plot(x[:limit_uncertainty], y1[:limit_uncertainty], 'r.', label='Beta Binomial1')
            plt.plot(x[:limit_uncertainty], y2[:limit_uncertainty], 'b.', label='Beta Binomial1')
            if scaling==True:
                scale = round(max(y3[:limit_uncertainty])/max(y2), -2)
                print(f'To improve data visualization 2*|true-pred|/(true+pred) has been scaled dividing by {scale}')
                print(f'uncertainty threshold = {threshold}')
                y3 = y3/scale
            
            plt.plot(x[uncertain_values], 
                     y3[uncertain_values], 
                     'ko', 
                     label='uncertain values')
            plt.plot(x[:limit_uncertainty], 
                     y3[:limit_uncertainty], 
                     'g.', 
                     label='2*|true-pred|/(true+pred)')
            plt.legend()
            
        return uncertain_values  
    
    def _betabinomial(self, x, C1, a1, b1):
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
    

    def define_flying_situation(self, 
                                uncertain_values: list):
        """
        Define the flying situation categories based on speed values.
        There is a range of speeds where it is not clear if the bird is flying 
        or not.
        Under the minimun value of that range we know it is stopped
        Over the maximun value of that range we know it is flying
        Inside that range we dont know
        Parameters:
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
        self.df['flying_situation'] = self.df.apply(partial_assign, axis=1)
        self.df['numerical_flying_situation'] = self.df.apply(self._assign_numerical_flying_categories, axis=1)
        
        return self.df
    
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
        
    def _assign_numerical_flying_categories(self, row: pd.Series):
        """
        Assign a numerical value 0, or 1 based on the nominal value
        of the column flying_situation.
        """
        if row['flying_situation'] == 'landed':
            return 0
        elif row['flying_situation'] == 'flying':
            return 1




class UndefinedFlyClassifier:
    """
    Create ML model to classify the undefined flying situations
    The model is trained with the previously flying situations defined
    Using the variable acc and acc_y the undefined values are classified with 
    a accuray>0.95 & cohen kappa score > 0.9
    """
    
    def __init__(self):
        pass
    
    
    def train_model(self,
                    df: pd.DataFrame,
                    scaler=MinMaxScaler(),
                    classifier=KNeighborsClassifier(n_neighbors=10)):
        """
        Train a classifier model and predict 'undefined' flying situations.
    
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the data for training and prediction.
    
        scaler : Scaler object, optional
            A data scaler to normalize the feature data.
            Defaults to MinMaxScaler().
    
        classifier : Classifier object, optional
            A classifier to fit and predict the model. 
            Defaults to KNeighborsClassifier(n_neighbors=10).
    
        Returns:
        --------
        pandas.DataFrame
            DataFrame with updated 'flying_situation' values, 
            replacing 'undefined' with predicted labels.
    
        Notes:
        ------
        This function trains a classifier model to predict 'undefined' 
        flying situations based on the provided data.
        The data is divided into a training set with 'flying' and 'landed' 
        situations and an undefined set whose flying situation is going to
        be predicted. 
        Prior to training and prediction, the model accuracy is check withe the 
        training data in the function: _check_model_accuracy
        Feature scaling is applied to both sets using the provided scaler. 
        The classifier is then trained on the training set and used to predict 
        the labels for the 'undefined' set.
        Predicted labels are assigned to the original DataFrame 'df',
        replacing the 'undefined' labels.
        The function returns the updated DataFrame with predicted flying situations.
        """
        self._check_model_accuracy(df, scaler, classifier)
    
        train_df = df[df['flying_situation'].isin(['flying', 'landed'])]
        undefined_df = df[df['flying_situation'] == 'undefined']
    
        X_train = train_df[['acc', 'acc_y']]
        Y_train = train_df['flying_situation']
    
        X_undefined = undefined_df[['acc', 'acc_y']]
    
        X_train_scaled = scaler.fit_transform(X_train)
        X_undefined_scaled = scaler.transform(X_undefined)
    
        classifier.fit(X_train_scaled, Y_train)
    
        predicted_labels = classifier.predict(X_undefined_scaled)
        df.loc[df['flying_situation'] == 'undefined', 'flying_situation'] = predicted_labels
        return df
    
   
    def _check_model_accuracy(self,
                              df : pd.DataFrame, 
                              scaler=MinMaxScaler(),
                              classifier=KNeighborsClassifier(n_neighbors = 10)):
        """
        Check the accuracy of a classifier model using a stratified K-fold cross-validation.
    
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the data for evaluation.
    
        scaler : Scaler object, optional
            A data scaler to normalize the feature data.
            Defaults to MinMaxScaler().
    
        classifier : Classifier object, optional
            A classifier to fit and predict the model. 
            Defaults to KNeighborsClassifier(n_neighbors=10).
    
        Returns:
        --------
        None
    
        Notes:
        ------
        The data is filtered to include only the 'flying' and 'landed' flying 
        situations, and then it is divided into features (X) and labels (Y).
        The function calculates and prints the average accuracy and kappa scores 
        based on the cross-validation results.
    
        The function does not return any values; it prints the results to the console.
        """
        filtered_df = df[df['flying_situation'].isin(['flying', 'landed'])]
    
        X = filtered_df[['acc', 'acc_y']]
        Y = filtered_df['flying_situation']
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
        results = [self._evaluate_single_split(split, X, Y, scaler, classifier)\
                   for split in kf.split(X, Y)]
        accuracy = [result[0] for result in results]
        kappa = [result[1] for result in results]
        print(f'Classifier accuracy: {np.mean(accuracy):.2f} \
                kappa: {np.mean(kappa):.2f}')
    
    def _evaluate_single_split(self,
                               split : tuple,
                               X : pd.DataFrame,
                               Y : pd.Series, 
                               scaler=MinMaxScaler(),
                               classifier=KNeighborsClassifier(n_neighbors=10)):
        """
        Evaluate a single split of training and testing data.
        
        Parameters:
        -----------
        split : tuple
            A tuple containing training and testing indices.
    
        X : pandas.DataFrame
            Feature data.
    
        Y : pandas.Series
            Target data.
    
        scaler : Scaler object
            A data scaler to transform the feature data.
    
        classifier : Classifier object
            A classifier to fit and predict the model.
    
        Returns:
        --------
        accuracy : float
        kappa : float
            Accuracy and kappa scores of the classifier.
        """
        X_train, X_test, Y_train, Y_test = self._select_split_data(split, X, Y)
        kappa, accuracy = self._evaluate_classifier(X_train, X_test, 
                                              Y_train, Y_test, 
                                              scaler, classifier)
        return kappa, accuracy
    
    def _evaluate_classifier(self,
                             X_train : pd.DataFrame,
                             X_test : pd.DataFrame,
                             Y_train : pd.DataFrame,
                             Y_test : pd.DataFrame, 
                             scaler = MinMaxScaler(),
                             classifier = KNeighborsClassifier(n_neighbors = 10)):
        """
        Evaluate a classifier using scaled training and testing data.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training feature data.
    
        X_test : pandas.DataFrame
            Testing feature data.
    
        Y_train : pandas.Series
            Training target data.
    
        Y_test : pandas.Series
            Testing target data.
    
        scaler : Scaler object
            A data scaler to transform the feature data.
    
        classifier : Classifier object
            A classifier to fit and predict the model.
    
        Returns:
        --------
        accuracy : float
        kappa : float
            Accuracy and kappa scores of the classifier.
        """
    
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        classifier.fit(X_train_scaled, Y_train)
        Y_pred = classifier.predict(X_test_scaled)
        
        accuracy = accuracy_score(Y_test, Y_pred)
        kappa = cohen_kappa_score(Y_test, Y_pred)
        
        return accuracy, kappa
    
    def _select_split_data(self, 
                           split : tuple,
                           X : pd.DataFrame,
                           Y : pd.Series):
        """
        Select data for a specific split defined by indices.
        
        Parameters:
        -----------
        split : tuple
            A tuple containing training and testing indices.
    
        X : pandas.DataFrame
            Feature data.
    
        Y : pandas.Series
            Target data.
    
        Returns:
        --------
        X_train : pandas.DataFrame
        X_test : pandas.DataFrame
        Y_train : pandas.Series
        Y_test : pandas.Series
            Training and testing feature data and target data.
        """
        train_index, test_index = split
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        return X_train, X_test, Y_train, Y_test
    

        
    
