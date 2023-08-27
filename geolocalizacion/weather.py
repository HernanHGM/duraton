# %% 1. IMPORTACINO LIBRERÍAS
import pandas as pd
import json
from datetime import datetime, timedelta

import requests
import os

from flask import Flask, jsonify
# %% 2. CREACION FUNCIONES

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
def get_weather(location='Zorita', start_date='2020-07-01', end_date='2023-01-01'):
    key = '12ee6540c60e4c5ca13175025231904'
    time_period = '1'
    location_format = location.replace(' ', '%20') + ',%20Spain'
    url = f'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key={key}&q={location_format}&format=json&date={start_date}&enddate={end_date}&tp={time_period}'
    response = requests.get(url)
    
    if response.ok:
        data = response.json()
        
        path = f'E:/duraton/geolocalizacion/_data/weather_data/{location}/'
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
def download_and_save_weather_data(location_list, start_date_dt, stop_date_dt, 
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
   


# Unused
def str_to_json(file_path):  
    with open(file_path, 'rb') as f:
        content = f.read()
    # Decodifica el contenido de bytes a una cadena
    content_str = content.decode('utf-8')
    data = json.loads(content_str)
    return data


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

def load_json_and_transformorm_to_dataframe(location_list):
    weather_data = {'hourly': {}, 'daily': {}}
    hourly_data = []
    daily_data = []
    for location in location_list:
        
        path = f'E:/duraton/geolocalizacion/_data/weather_data/{location}/'
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

def save_weather_dataframe(weather_dict):
    folder_path = 'E:/duraton/geolocalizacion/_data/weather_data/all_locations/'
    for time_period, dataframe in weather_dict.items():
        file_name = f'{time_period}.csv'
        file_path = os.path.join(folder_path, file_name)
        dataframe.to_csv(file_path)
        
def load_weather_dataframe():
    directory = 'E:/duraton/geolocalizacion/_data/weather_data/all_locations'
    weather_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            csv_data = pd.read_csv(filepath, index_col=0)
            filename = filename.replace('.csv', '')
            weather_dict[filename] = csv_data
    return weather_dict