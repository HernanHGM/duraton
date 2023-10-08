import math
import numpy as np
import pandas as pd
import rasterio
from pandasql import sqldf

class ElevationLoader():
    '''
    Loads terrain data stored in files that cotains tiles of 1º*1º
    Example: N40W006 contains from N39.001 to N40 and W5.001 to W6 
    (west is negative longitude) 
    So the file information is defined by the left bottom corner
        -left--> (floor(longitude))
        -bottom--> (floor(latitude))
    '''
    
    def __init__(self, 
                 base_path = 'E:\\duraton\\geolocalizacion\\_data\\elevation\\raw'):
        self.base_path = base_path
     
    def load_necessary_files(self, df_fly):
        '''
        Given a dataframe with bird geopositions loads the necessary terrain 
        tiles containing the terrain elevation of the area travelled by the bird

        Parameters
        ----------
        df_fly : pd.DataFrame
            Bird dataframe with geopositions.

        Returns
        -------
        df_elevation: pd.DataFrame
            DataFrame que contiene los datos de elevación y los límites geográficos de cada casilla.
    
            El DataFrame resultante tiene las siguientes columnas:
            
            - 'elevation': Datos de elevación de cada casilla.
            - 'min_long': Longitud correspondiente al límite izquierdo de cada casilla.
            - 'max_long': Longitud correspondiente al límite derecho de cada casilla.
            - 'min_lat': Latitud correspondiente al límite inferior de cada casilla.
            - 'max_lat': Latitud correspondiente al límite superior de cada casilla.

        '''
        max_lat, min_lat, max_long, min_long = self._find_coordinates_extremes(df_fly)        
        corners = self._create_corners_list(max_lat, min_lat, max_long, min_long)
        filenames_list = [self._create_coordinates_filename(*corner) for corner in corners]
        complete_filenames = ['\\'.join([self.base_path, filename]) for filename in filenames_list]
        df_list = [self._load_hgt_file(file_path) for file_path in complete_filenames]
        df_elevation = pd.concat(df_list)
        
        df_elevation = self._remove_unnecessary_data(df_fly, df_elevation)
        return df_elevation
    
    
    def _create_coordinates_filename(self, max_lat: int, min_long: int):
        '''
        Given the coordinates of a tile terrain the left bottom corner, 
        generates the name of the file to load.    
    
        Parameters
        ----------
        max_lat : int
            max latitude of the tile terrain that is going to be load.
        min_long : int
            min longitud of the tile terrain that is going to be load.
    
        Returns
        -------
        file_name: str
            name of the file to load.
    
        '''
        if max_lat>=0:
            hemisphere='N'
        else:
            hemisphere='S'
    
        if min_long>=0:
            side='E'
        else:
            side='W'
        lat_str = str(abs(max_lat)).zfill(2)
        long_str = str(abs(min_long)).zfill(3)
        file_name = ''.join([hemisphere, lat_str, side, long_str, '.hgt'])
        return file_name

    def _find_coordinates_extremes(self, df_fly: pd.DataFrame):
        '''
        Get the latitude extremes of the dataframe an round them up (floor())
        Get the longitude extremes of the dataframe an round them down (floor())
        
    
        Parameters
        ----------
        df_fly : pd.DataFrame
            Bird dataframe with geopositions.
    
        Returns
        -------
        max_lat : int
            max latitude reached by the bird rounded up to arcdegrees.
        min_lat : int
            min latitude reached by the bird rounded up to arcdegrees.
        max_long : int
            max longitude reached by the bird rounded down to arcdegrees.
        min_long : int
            min longitude reached by the bird rounded down to arcdegrees.
    
        '''
        max_lat = math.floor(max(df_fly.Latitude))
        min_lat = math.floor(min(df_fly.Latitude))
        max_long = math.floor(max(df_fly.Longitude))
        min_long= math.floor(min(df_fly.Longitude))
        return max_lat, min_lat, max_long, min_long

    def _create_corners_list(self,
                             max_lat: int, min_lat: int, 
                             max_long: int, min_long: int):
        '''
        Receives the extremes coordinates of a bird displacement and generates
        a list containg tuples of the terrain tiles needed to cover the surface
        travelled by the bird.
        We have the terrain data stored in files that cotains tiles of 1º*1º
        Example: N40E006 contains from N39.001 to N40 and E6.001 to E7 
        So the file information is defined by the left bottom corner
            -left--> (floor(longitude))
            -bottom--> (floor(latitude))
    
        Parameters
        ----------
        max_lat : int
            max latitude reached by the bird rounded up to arcdegrees.
        min_lat : int
            min latitude reached by the bird rounded up to arcdegrees.
        max_long : int
            max longitude reached by the bird rounded down to arcdegrees.
        min_long : int
            min longitude reached by the bird rounded down to arcdegrees.
    
        Returns
        -------
        corners : list
            List containing tuples of two integers that represents bottom left corners
            of terrain tiles.
    
        '''    
        
        lat_list = np.arange(min_lat, max_lat+1)
        long_list = np.arange(min_long, max_long+1)
    
        # list containing al tiles corners
        corners = [(lat, long) for lat in lat_list for long in long_list]
        return corners


 
    def _load_hgt_file(self, file_path):
        """
        Carga un archivo de formato HGT y convierte los datos de elevación en un DataFrame.
    
        Parameters
        ----------
        file_path : str
            Ruta del archivo HGT que se desea cargar.
    
        Returns
        -------
        pd.DataFrame
            DataFrame que contiene los datos de elevación y los límites geográficos de cada casilla.
    
            El DataFrame resultante tiene las siguientes columnas:
            
            - 'elevation': Datos de elevación de cada casilla.
            - 'min_long': Longitud correspondiente al límite izquierdo de cada casilla.
            - 'max_long': Longitud correspondiente al límite derecho de cada casilla.
            - 'min_lat': Latitud correspondiente al límite inferior de cada casilla.
            - 'max_lat': Latitud correspondiente al límite superior de cada casilla.
            
        Ejemplo
        -------
        file_path = 'E:\\duraton\\geolocalizacion\\_data\\elevation\\raw\\N39W006.hgt'
        elevation_data_df = load_hgt_file(file_path)
        """
        # Abrir el archivo HGT usando rasterio
        with rasterio.open(file_path) as dataset:
            # Leer los datos de elevación de la banda 1
            elevation_array = dataset.read(1)
    
            # Obtener los límites geográficos del archivo HGT
            lon_min, lat_max = np.round(dataset.bounds.left, 2), np.round(dataset.bounds.top, 2)
            lon_max, lat_min = np.round(dataset.bounds.right, 2), np.round(dataset.bounds.bottom, 2)
    
            # Calcular el paso de longitud y latitud entre cada casilla
            lon_step = (lon_max - lon_min) / (elevation_array.shape[1])
            lat_step = (lat_max - lat_min) / (elevation_array.shape[0])
            # Generar las coordenadas de latitud y longitud para cada casilla
            latitudes = np.linspace(lat_max, lat_min, elevation_array.shape[0], endpoint=False)
            longitudes = np.linspace(lon_min, lon_max, elevation_array.shape[1], endpoint=False)
    
            # Ravelizar los datos de elevación y generar las coordenadas de los límites geográficos
            elevation_data = elevation_array.ravel()
            left_bounds = np.tile(longitudes, elevation_array.shape[0])
            right_bounds = np.tile(longitudes + lon_step, elevation_array.shape[0])
            top_bounds = np.repeat(latitudes, elevation_array.shape[1])
            bottom_bounds = np.repeat(latitudes - lat_step, elevation_array.shape[1])
    
            # Crear el DataFrame con los datos de elevación y límites geográficos
            elevation_df = pd.DataFrame({
                'elevation': elevation_data,
                'min_long': left_bounds,
                'max_long': right_bounds,
                'min_lat': bottom_bounds,
                'max_lat': top_bounds
            })
    
        return elevation_df
    
    def _remove_unnecessary_data(self,
                                 df_fly: pd.DataFrame, 
                                 df_elevation: pd.DataFrame):
        '''
        Due to the huge amount of data contained in df_elevation, a filter to 
        reduced the unused data is vital to reduce execution time.
        In df_elevation all the data stored in terrain tiles is loaded, 
        but only the terrain information where the bird flies is needed.

        Parameters
        ----------
        df_fly : pd.DataFrame
            Bird dataframe with geopositions.
        df_elevation : pd.DataFrame
            Terrain dataframe with elevation terrain.

        Returns
        -------
        df_elevation : pd.DataFrame
            Same df_elevation but filtered

        '''
        max_lat = max(df_fly.Latitude)
        min_lat = min(df_fly.Latitude)
        max_long = max(df_fly.Longitude)
        min_long= min(df_fly.Longitude)
        condition = (df_elevation.max_lat <= max_lat) &\
                    (df_elevation.min_lat >= min_lat) &\
                    (df_elevation.max_long <= max_long) &\
                    (df_elevation.min_long >= min_long)
        df_elevation = df_elevation.loc[condition]
        return df_elevation


class FlyElevationJoiner():
    '''
    Joins df_fly and df_elevation
    '''
    def __init__(self):
        pass
        
    def fly_elevation_join(self, 
                           df_fly: pd.DataFrame, 
                           df_elevation: pd.DataFrame):
        '''
        As the bird geoposition refers to exact space points but elevation 
        position refers to tiles of 1"*1" the merge is conditional so the
        difficulty is smaller if is done by sql command.

        Parameters
        ----------
        df_fly : pd.DataFrame
            Bird fly dataframe
        df_elevation : pd.DataFrame
            Elevation terrain dataframe.

        Returns
        -------
        df_joined : pd.DataFrame
            Bird fly dataframe sit terrain elevation merged.
            
        Example
        -------
        # DataFrame con coordenadas geográficas concretas
        df_coordenadas = pd.DataFrame({
            'latitud': [39.5684, 40.1234, 38.8765],
            'longitud': [-5.74272, -6.9876, -4.5678],
           })

        # DataFrame con rangos
        df_rangos = pd.DataFrame({
            'rango_lat_min': [39.0, 40.0, 38.5],
            'rango_lat_max': [39.9, 40.5, 39.0],
            'rango_lon_min': [-6.0, -7.0, -6.0],
            'rango_lon_max': [-5.5, -6.5, -4.0],
            'valor_rango': [10, 20, 30]  # Datos adicionales que quieres conservar del segundo DataFrame
        })

        # Consulta SQL para realizar la unión
        query = """
        SELECT *
        FROM df_coordenadas AS c
        LEFT JOIN df_rangos AS r
        ON c.latitud BETWEEN r.rango_lat_min AND r.rango_lat_max
        AND c.longitud BETWEEN r.rango_lon_min AND r.rango_lon_max
        """
        import pandas as pd
        from pandasql import sqldf

        # Ejecutar la consulta usando pandasql
        df_resultado = sqldf(query, locals())

        print(df_resultado)
        '''
        
        # Consulta SQL para realizar la unión
        query = """
        SELECT *
        FROM df_fly AS df1
        LEFT JOIN df_elevation AS df2
        ON df1.Latitude BETWEEN df2.min_lat AND df2.max_lat
        AND df1.Longitude BETWEEN df2.min_long AND df2.max_long
        """

        # Ejecutar la consulta usando pandasql
        df_joined = sqldf(query, locals())
        df_joined = self._clean(df_joined)
        return df_joined
        
    def _clean(self, df_joined):
        columns = df_joined.columns.to_list()
        unnamed_cols = [col for col in columns if 'UTC' in col]
        drop_cols = ['ID', 'datatype', 'hdop',
                     'min_long', 'max_long',
                     'min_lat', 'max_lat'] + unnamed_cols
        df_joined = df_joined.drop(labels=drop_cols, axis=1)
        df_joined['bird_altitude'] = df_joined.Altitude_m - df_joined.elevation
        return df_joined

