import pandas as pd
import numpy as np

class Cleaner:
    
    def __init__(self, path):
        """
        Inicializa la clase y carga el archivo excel especificado en el argumento file_path.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo csv a cargar en el dataframe.
        """
        self.df = pd.read_excel(path)

    def _force_numeric(self):
        """
        Convierte todas las columnas no numéricas del dataframe en numéricas, excepto las columnas "edad", "sexo" y "especie".
        """
        cols = [col for col in self.df.columns if col not in ["edad", "sexo", "especie"]]
        self.df[cols] = self.df[cols].apply(pd.to_numeric, errors='coerce')               

    def select_columns(self, columns=None):
        """
        Selecciona solo las columnas especificadas en el argumento columns. Si no se especifica ningún valor en columns,
        se seleccionan las columnas "edad", "sexo", "especie", "cola", "izda L", "izda DV", "dcha L" y "dcha DV".
        
        Parámetros:
        -----------
        columns : list, opcional
            Lista de nombres de columnas a seleccionar. Por defecto es None, lo que selecciona las columnas predefinidas.
        """
        if columns is None:
            columns = ["especie", "edad", "sexo" ,
                       'peso', 'antebrazo', 'clave',
                       "izda L", "izda DV", "dcha L", "dcha DV",
                       "cola", 'rectrix c',
                       'ancho ala', 'ala d', 'ala v',	
                       '7º  1ª', 'cañón en 7ª',
                       'envergadura', 'longitud total',	
                       'long pico', 'alto pico', 'ancho pico',	
                       'long cabeza', 'ancho cabeza']
        self.df = self.df[columns]
        self._force_numeric()
        
    def select_rows(self, edad, especie, sexo):
        """
        Selecciona solo las filas en las que se cumplan las condiciones especificadas en los argumentos edad, especie y sexo.
        
        Parámetros:
        -----------
        edad : list
            Lista de edades a seleccionar.
        especie : str
            Valor de especie a seleccionar.
        sexo : list
            Lista de valores de sexo a seleccionar.
        """
        edad_condition = self.df['edad'].isin(edad)
        especie_condition = self.df['especie'] == especie
        sexo_condition = self.df['sexo'].isin(sexo)
        # query = f"({edad_condition}) and ({especie_condition}) and ({sexo_condition})"
        # self.df = self.df.query(query)
        all_conditions = edad_condition & especie_condition & sexo_condition
        self.df = self.df[all_conditions]
        

               
        
    def remove_outliers(self):
        """
        Elimina los outliers de todas las columnas numéricas del dataframe.
        """
        Q1 = self.df.quantile(0.05, numeric_only = True)
        Q3 = self.df.quantile(0.95, numeric_only = True)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        for col in self.df.columns:
            if self.df[col].dtype != "object":
                self.df = self.df[(self.df[col] >= lower_bound[col]) & (self.df[col] <= upper_bound[col])]
        
    def calculate_means(self):
        """
        Calcula la media entre las columnas "izda L" y "dcha L" y crea una nueva columna "L_media".
        Calcula la media entre las columnas "izda DV" y "dcha DV" y crea una nueva columna "DV_media".
        Elimina las columnas 'izda L', 'izda DV', 'dcha L', 'dcha DV'
        """
        self.df['L_media'] = self.df[['izda L', 'dcha L']].mean(skipna=True, axis=1)
        self.df['DV_media'] = self.df[['izda DV', 'dcha DV']].mean(skipna=True, axis=1)
        self.df = self.df.drop(['izda L', 'izda DV', 'dcha L', 'dcha DV'], axis=1)

    def remove_empty_columns(self, tolerance=0.1):
        """
        Elimina columnas cuyo ratio de vacío (emptiness) es mayor que la tolerancia
        especificada.
        Elimina filas del dataframe que contienen al menos un valor nulo (NaN),
    
        Parámetros:
        -----------
        tolerance : float
            Valor entre 0 y 1 que indica el umbral de tolerancia para el ratio de vacío.
    
        """
        # Calcula el ratio de vacío para cada columna
        n = len(self.df)
        emptiness = self.df.isnull().sum() / n
        # Elimina las columnas cuyo ratio de vacío es mayor que la tolerancia
        cols_to_drop = emptiness[emptiness > tolerance].index
        self.df = self.df.drop(cols_to_drop, axis=1)
        # Elimina las filas que contienen valores nulos
        self.df = self.df.dropna()
   
    def calculate_correlation(self):
        """
        Calcula la correlacion entre todas las variables numericas y el sexo
        
        Returns:
        -------
        corr : DataFrame
            Un DataFrame que contiene la matriz de correlación entre las variables numericas y el sexo
        """
        # Elimina las columnas "edad" y "especie" del DataFrame original
        corr_df = self.df.drop(["edad", "especie"], axis=1)
        
        # Reemplaza los valores de "sexo" con 1 para "Macho" y 0 para "Hembra"
        corr_df['sexo'] = corr_df['sexo'].replace({'Macho':1, 'Hembra':0})
        
        # Calcula la matriz de correlación
        corr = corr_df.corr()
        
        # Devuelve el DataFrame con la matriz de correlación
        return corr

