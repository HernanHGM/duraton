from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

class FeatureScaler:
    def __init__(self, df):
        self.df = df
        self.label = None
        self.df_encoded = None
        self.df_scaled = None
    
    def label_encoder(self, label='sexo'):
        """
        Codifica la variable categórica label utilizando 
        LabelEncoder de sklearn.preprocessing.
        
        Parámetros:
        -----------
        label : string
            variable a codificar.
        """
        self.label = label
        # Crear una copia del DataFrame original para evitar modificar los datos originales
        self.df_encoded = self.df.copy()      
        # Aplicar la codificación a la columna 'sexo'
        self.df_encoded[self.label] = LabelEncoder.fit_transform(self.df_encoded[self.label])

    
    def scale_numeric_variables(self, scaling_type='min_max'):
        """
        Escala las variables numéricas del DataFrame utilizando 
        el método de escalamiento especificado.
        
        Parámetros:
        -----------
        scaling_type : str, opcional (por defecto='min_max')
            Tipo de escalamiento a aplicar. Puede ser uno de los siguientes valores: 'min_max', 'z_score'.
        """
        # Crear una copia del DataFrame original para evitar modificar los datos originales
        self.df_scaled = self.df_encoded.copy()
        
        # Crear un diccionario para almacenar los métodos de escalamiento
        scaling_methods = {'min_max': MinMaxScaler(),
                           'z_score': StandardScaler()}
        
        # Obtener el método de escalamiento correspondiente
        scaler = scaling_methods.get(scaling_type)
        # Selecciona todas las variables salvo la etiqueta
        features= [col for col in self.df.columns if col != self.label]
        #Escalara las variables
        self.df_scaled[features] = scaler.fit_transform(self.df_scaled[features])
        


