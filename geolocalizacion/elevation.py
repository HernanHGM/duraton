import numpy as np
import pandas as pd
import rasterio


def load_hgt_file(file_path):
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



