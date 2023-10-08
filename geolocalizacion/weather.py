# %% 1. IMPORTACION LIBRERÍAS
import pandas as pd
import json
from datetime import datetime, timedelta
from geopy.distance import geodesic

import requests
import os

from flask import Flask, jsonify
# %% 2. CREACION FUNCIONES

# =============================================================================
# DOWNLOAD AND SAVE WEATHER DATA
# =============================================================================
def _save_data_to_json(data, filepath):
    # Crear directorio si no existe
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Guardar los datos en formato JSON
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)
        
        
app = Flask(__name__)

@app.route('/api/get_weather')
def get_weather(location: str,
                start_date: str='2020-01-01',
                end_date: str='2023-01-01'):
    key = 'c5183e6cb96f4b8190a103026230309'
    time_period = '1'
    location_format = location.replace(' ', '%20') + ',%20Spain'
    url = f'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key={key}&q={location_format}&format=json&date={start_date}&enddate={end_date}&tp={time_period}'
    response = requests.get(url)
    
    if response.ok:
        data = response.json()
        
        path = f'E:/duraton/geolocalizacion/_data/weather/{location}/'
        file = f'{start_date}.json'
        file_path = os.path.join(path, file)
        
        _save_data_to_json(data, file_path)
    else:
        raise ValueError('No se han descargado los datos')

    return data



def _get_new_date(data):
    # Obtener la última fecha
    last_date = data['data']['weather'][-1]['date']

    # Crear una variable new_date con un día más que la última fecha
    last_date_datetime = datetime.strptime(last_date, '%Y-%m-%d')
    new_date_datetime = last_date_datetime + timedelta(days=1)
    new_date = new_date_datetime.strftime('%Y-%m-%d')
    
    return new_date    

#### Revisar
def download_and_save_weather_data(location_list, 
                                   start_date_dt, stop_date_dt, 
                                   start_date, original_date):
    for location in location_list:
        print(location, '---------------------------------------------')
        while(start_date_dt < stop_date_dt):
            data = get_weather(location=location, start_date=start_date)
            start_date = _get_new_date(data) 
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            print(start_date)
            
        start_date = original_date
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
   
def load_json_and_transformorm_to_dataframe(location_list):
    weather_data = {'hourly': {}, 'daily': {}}
    hourly_data = []
    daily_data = []
    for location in location_list:
        
        path = f'E:/duraton/geolocalizacion/_data/weather/{location}/'
        data_by_location = _load_all_json_files_in_directory(path)
        
        dfs_hourly = [_json_to_pandas(data)[0] for data in data_by_location]
        dfs_daily = [_json_to_pandas(data)[1] for data in data_by_location]
        
        # Concatenar todos los DataFrames en uno solo
        merged_hourly = pd.concat(dfs_hourly)
        merged_daily = pd.concat(dfs_daily)
        
        # Agregar columna "location" con los valores correspondientes
        merged_hourly['location'] = location
        merged_daily['location'] = location
        
        hourly_data.append(merged_hourly)
        daily_data.append(merged_daily)
    
     
    weather_data['hourly'] =  pd.concat(hourly_data).reset_index(drop=True)
    weather_data['daily'] = pd.concat(daily_data).reset_index(drop=True)
    return weather_data

def _load_data_from_json(filepath):
    # Abrir el archivo JSON
    with open(filepath) as infile:
        # Cargar los datos desde el archivo
        data = json.load(infile)
    
    # Devolver los datos cargados
    return data

def _load_all_json_files_in_directory(directory):
    json_data_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            json_data = _load_data_from_json(filepath)
            json_data_list.append(json_data)
    return json_data_list

def _format_hourly_dataframe(df):
    df = df.astype({'time': 'int',
                                  'tempC': 'float16',
                                  'tempF': 'float16',
                                  'windspeedMiles': 'float16',
                                  'windspeedKmph':'float16',
                                  'winddirDegree': 'float16',
                                  'winddir16Point': 'object',
                                  'weatherCode': 'float16',
                                  'weatherIconUrl': 'object',
                                  'weatherDesc': 'object',
                                  'precipMM': 'float16',
                                  'precipInches': 'float16',
                                  'humidity': 'float16',
                                  'visibility': 'float16',
                                  'visibilityMiles': 'float16',
                                  'pressure': 'float16',
                                  'pressureInches': 'float16',
                                  'cloudcover': 'float16',
                                  'HeatIndexC': 'float16',
                                  'HeatIndexF': 'float16',
                                  'DewPointC': 'float16',
                                  'DewPointF': 'float16',
                                  'WindChillC': 'float16',
                                  'WindChillF': 'float16',
                                  'WindGustMiles': 'float16',
                                  'WindGustKmph': 'float16',
                                  'FeelsLikeC': 'float16',
                                  'FeelsLikeF': 'float16',
                                  'uvIndex': 'float16',
                                  'date': 'datetime64[ns]'
                                  })
    df.time = (df.time/100) #Vienen 0,100,200... 2300
    df.time = pd.to_datetime(df.time, format='%H').dt.time
    df.date = df.date.dt.date
    return df
    

def _json_to_pandas(data):
    df_hourly = pd.json_normalize(data['data']['weather'], 
                                   record_path=['hourly'], 
                                   meta=[['date']])
    df_hourly = _format_hourly_dataframe(df_hourly)
    
    
    
    df_daily = pd.json_normalize(data['data']['weather'], 
                               record_path=['astronomy'], 
                               meta=[['date'], ['maxtempC'], 
                                     ['mintempC'], ['avgtempC'],
                                     ['sunHour'], ['totalSnow_cm'],
                                     ['uvIndex']])
    return df_hourly, df_daily



def save_weather_dataframe(weather_dict):
    folder_path = 'E:/duraton/geolocalizacion/_data/weather/all_locations/'
    for time_period, dataframe in weather_dict.items():
        file_name = f'{time_period}.csv'
        file_path = os.path.join(folder_path, file_name)
        dataframe.to_csv(file_path)

# =============================================================================
# JOIN WEATHER AND FLYING DATA
# =============================================================================
def load_weather_dataframe(directory: str='E:/duraton/geolocalizacion/_data/weather/all_locations'):
    '''
    loads csv directory and generates dataframes that are stored in a directory
    The three dataframe are merged into a complete dataframe.
    To load all files automatically: 
        filenames_list = os.listdir(directory)
        if filename.endswith('.csv'):
        filename = filename.replace('.csv', '')

    Parameters
    ----------
    directory : str, optional
        directory were daily, hourly weather from villages and its coordinates 
        are sotred in three dataframes. 
        The default is 'E:/duraton/geolocalizacion/_data/weather/all_locations'.

    Returns
    -------
    weather_dict : dict
        dictionary with three dataframes: 
            village coordinates dataframe.
            village hourly weather dataframe
            village daily weather dataframe
    '''
    weather_dict = {}
    filenames_list = ['hourly', 'daily', 'coordinates']
    for filename in filenames_list:
            filepath = os.path.join(directory, '.'.join([filename,'csv']))
            csv_data = pd.read_csv(filepath, index_col=0)
            weather_dict[filename] = csv_data
    return weather_dict

def find_nearest_location(row: pd.Series, locations_df: pd.DataFrame):
    '''
    Given a row of a dataframe containing a Latitude and Longitude column
    and a Dataframe containing several locations with their respective
    Latitudes and longitudes, returns the name of the closest location.

    Parameters
    ----------
    row : pd.Series
        row of the dataframe whose closest location we want to know.
        Mainly the dataframe contains flying bird data
    locations_df : pd.DataFrame
        Dataframe with several locations and their latitude, longitude and 
        Altitude.

    Returns
    -------
    closest_location : str
        Name of the closest location to the row.

    '''
    location_coords = list(zip(locations_df['Latitude'], 
                                locations_df['Longitude']))
    distances = [geodesic((row['Latitude'], row['Longitude']), loc).kilometers\
                  for loc in location_coords]
    nearest_index = distances.index(min(distances))
    closest_location = locations_df.index.tolist()[nearest_index]
    return closest_location

def join_fly_weather(weather_dict, df_fly, freq: str):
    """
    Combina datos de vuelo y datos meteorológicos según la frecuencia especificada.

    Parámetros:
    -----------
    weather_dict : dict
        Un diccionario que contiene los datos meteorológicos agrupados por frecuencia ('daily' o 'hourly').

    df_fly : pandas.DataFrame
        El DataFrame con los datos de vuelo.

    freq : str
        La frecuencia de los datos a combinar. Puede ser 'daily' o 'hourly'.

    Returns:
    --------
    pandas.DataFrame, list
        Un DataFrame resultante con los datos de vuelo y meteorológicos combinados,
        y una lista de las variables meteorológicas que se han incluido en la combinación.

    Notas:
    ------
    Esta función combina datos de vuelo y datos meteorológicos según la frecuencia especificada ('daily' o 'hourly').
    Los datos meteorológicos deben estar contenidos en el diccionario 'dict_weather', donde las claves son las
    frecuencias ('daily' o 'hourly') y los valores son los DataFrames correspondientes.

    Para cada frecuencia, se seleccionan diferentes variables meteorológicas para combinar con los datos de vuelo.
    El resultado de la combinación se realiza utilizando la función merge de pandas y los parámetros de unión
    ('left_on' y 'right_on') se determinan en función de la frecuencia.

    La función devuelve el DataFrame resultante con los datos combinados y una lista de las variables meteorológicas
    que se han incluido en la combinación.
    """
    df_weather = weather_dict[freq]
    
    if freq == 'hourly':
        weather_variables = ['tempC', 'DewPointC', 
                             'windspeedKmph', 'pressure', 'visibility', 
                             'cloudcover', 'precipMM', 'humidity']
        fly_merge_variables = ['UTC_date', 'hour', 'closest_location']
        weather_merge_variables = ['date', 'time', 'location']
    
    if freq == 'daily':   
        totalSunHour = (pd.to_datetime(df_weather['sunset']) -
                        pd.to_datetime(df_weather['sunrise'])).dt.seconds / 3600
        df_weather = df_weather.assign(totalSunHour=totalSunHour)
        weather_variables = ['maxtempC', 'mintempC', 'avgtempC', 
                             'sunHour', 'totalSunHour', 'uvIndex']
        fly_merge_variables = ['UTC_date', 'closest_location']
        weather_merge_variables = ['date', 'location']
    
    weather_selected_variables = weather_variables + weather_merge_variables
    data_joined = df_fly.merge(df_weather[weather_selected_variables], 
                               left_on=fly_merge_variables,
                               right_on=weather_merge_variables, 
                               how='left')
    data_joined = data_joined.drop(weather_merge_variables, axis=1)
    return data_joined, weather_variables


# =============================================================================
# UNUSED FUNCTIONS
# =============================================================================
def str_to_json(file_path):  
    with open(file_path, 'rb') as f:
        content = f.read()
    # Decodifica el contenido de bytes a una cadena
    content_str = content.decode('utf-8')
    data = json.loads(content_str)
    return data

def _join_weather_data(df_hourly: pd.DataFrame,
                       df_daily: pd.DataFrame,
                       df_coordinates: pd.DataFrame):
    '''
    Join in one single dataframe the weather data with hourly and daily 
    frequency, and the localizations geopositions. Hourly and daily dataframes
    contains not only different frequency but different variables

    Parameters
    ----------
    df_hourly : pd.DataFrame
        weather data with hourly frequency.
    df_daily : pd.DataFrame
        weather data with daily frequency.
    df_coordinates : pd.DataFrame
        Latitude, longitude and altitude of each localization.

    Returns
    -------
    df_complete : pd.DataFrame
        dataframe with the houly, daily and geoposition information joined.

    '''
    df_weather = df_hourly.merge(df_daily,
                                 on = ['date', 'location'])
    df_complete = df_weather.merge(df_coordinates, 
                                   left_on = 'location',
                                   right_index=True)
    return df_complete

