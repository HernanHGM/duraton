from sklearn.preprocessing import LabelEncoder
import sys
path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)
    
from sexaje import parameters

class Preprocessor:
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
        self.df_encoded[self.label] = LabelEncoder().fit_transform(self.df_encoded[self.label])

    
    def feature_scaler(self, scaling_type='min_max'):
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
        
        # Obtener el método de escalamiento correspondiente
        scaler = parameters.scaler_dict[scaling_type]
        # Selecciona todas las variables salvo la etiqueta
        features= [col for col in self.df.columns if col != self.label]
        #Escalara las variables
        self.df_scaled[features] = scaler.fit_transform(self.df_scaled[features])
        


