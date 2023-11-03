"""
Created on Thu Jan  5 12:50:28 2023

@author: Hernán García Mayoral

Module of functions dedicated to loaad clean and preprocess data
"""
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from typing import List

# =============================================================================
# IMPORTACION DATOS
# =============================================================================
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def load_data(path: str, 
              filenames: List[str], 
              pre_process: bool=True,
              reindex_data: bool=True,
              freq: int=5):
    """
    Load and process data from multiple CSV files and return a combined DataFrame.
    
    Parameters:
    -----------
    path : str
        The path to the directory containing CSV files.
    
    filenames : List[str]
        A list of file names inside the directory provided by path
        to be loaded and processed.
    
    pre_process : bool, optional
        Whether to perform data pre-processing. Default is True.
    
    reindex_data : bool, optional
        Whether to reindex and interpolate the data. Default is True.
    
    freq : int, optional
        The frequency for reindexing and interpolation expressed in minutes.
        Default is 5.
    
    Returns:
    --------
    pandas.DataFrame
        A combined DataFrame containing data from multiple CSV files.
    """
    # join root_path and filenames into full path
    full_filenames = (path + '\\' + name for name in filenames) 
    
    li = []
    date_columns = ['UTC_datetime']
    for name in full_filenames:
        print(name)
        df = pd.read_csv(name, 
                         parse_dates = date_columns, 
                         index_col=False,
                         encoding="ISO-8859-1")
        df = _clean_unnamed(df)
        print('N registers: ', df.shape[0])
        if pre_process == True:
            df.rename(columns={'device_id': 'ID'}, inplace=True)
            df = _remove_outliers(df, columns=['Latitude', 'Longitude'])
            df = df.loc[df.Altitude_m > 0]
        if reindex_data == True:
            df = reindex_interpolate(df, freq = freq)
            print(freq)
        if pre_process == True:
            df = enriquecer(df, filenames)
        li.append(df)
    
    df = pd.concat(li, axis=0, ignore_index=True)
    return df

def _clean_unnamed(df):
    columns = df.columns.to_list()
    unnamed_cols = [col for col in columns if 'Unnamed' in col]
    df = df.drop(labels=unnamed_cols, axis=1)
    return df

def enriquecer(df : pd.DataFrame, 
                filenames: List[str]):
    # Añadimos la columna de hora
    df = df.assign(hour=df.UTC_datetime.dt.strftime('%H'))
    df['hour']= pd.to_datetime(df['hour'], format='%H').dt.time.astype(str)
    df['UTC_datetime_pre'] = df['UTC_datetime'].shift(-1)
    df['time_step_s'] = (df['UTC_datetime_pre']-df['UTC_datetime']).dt.seconds
    df['month_name'] = pd.to_datetime(df['UTC_date'], format="%Y/%m/%d")\
                         .dt.strftime('%B')
    df['month_number'] = pd.to_datetime(df['UTC_date'], format="%Y/%m/%d")\
                         .dt.strftime('%m')
    df['week_number'] = pd.to_datetime(df['UTC_date'], format="%Y/%m/%d")\
                          .dt.strftime('%W')
                          
    df['breeding_period'] = df['UTC_datetime'].apply(_categorize_breeding_period)
                          
    # corremos una posicion los datos de posicion
    df['Latitude_lag'] = df['Latitude'].shift(-1)
    df['Longitude_lag'] = df['Longitude'].shift(-1)
    df['Altitude_m_lag'] = df['Altitude_m'].shift(-1)
    # Calculamos distancias recorridas usando datos de posicion
    df = calculate_distance_intra_df(df,
                                     'Latitude_lag', 'Latitude',
                                     'Longitude_lag', 'Longitude',
                                     'Altitude_m_lag', 'Altitude_m')   

    df['acc'] = np.sqrt(df.acc_x**2 + df.acc_y**2 + df.acc_z**2)
    df['mag'] = np.sqrt(df.mag_x**2 + df.mag_y**2 + df.mag_z**2)
    
    info_archivos = list(map(extract_info, filenames))
    info_pajaros = pd.DataFrame(info_archivos, columns=['specie','ID','name'])
    info_pajaros['color'] = pd.Series(['green', 'blue', 'purple', 'red', 'orange'])
    df = df.merge(info_pajaros, how = 'left', on = 'ID')

    return df

def _categorize_breeding_period(row : pd.Series):
    '''
    March, april, may, june and july compose the breeding period
    The rest of months are no breeding perior

    Parameters
    ----------
    row : pd.Series
        UTC_datetime column that contains the month of the data.

    Returns
    -------
    period_value : str
        breeding or no breeding period.

    '''
    if 3 <= row.month <= 7:  # Months from March to July
        period_value = 'breeding'
    else:
        period_value = 'no breeding'
    return period_value


def extract_info(nombre_archivo):
    nombre_limpio = nombre_archivo.replace('.csv', '')
    partes = nombre_limpio.split('_')
    
    conversor_especie={'Aquada': 'Aquila adalberti',
                        'Aquchr': 'Aquila chrisaetos',
                        'Aqufas': 'Aquila fasciata'}
    
    especie = conversor_especie[partes[0]]
    ID = int(partes[1])
    nombre = partes[2]
    return especie, ID, nombre
# =============================================================================
# LIMPIEZA
# =============================================================================
def _remove_outliers(data, columns = None):

    data_o = data.copy()
    if columns != None:
        data = data[columns]
        
    lim_df = data.quantile((0.05, 0.95)).transpose()
    lim_df['IQR'] = lim_df[0.95] - lim_df[0.05]
    lim_df['Upper'] = lim_df[0.95] + lim_df['IQR']
    lim_df['Lower'] = lim_df[0.05] - lim_df['IQR']
    lim_df = lim_df.transpose()
    
    up_condition = (data <= lim_df.loc['Upper',:])
    low_condition = (data >= lim_df.loc['Lower',:])
    condition = up_condition & low_condition
    
    data_clean = data_o[condition.all(axis=1)]
    
    return data_clean

# =============================================================================
# INTERPOLATE
# =============================================================================
def reindex_interpolate(df : pd.DataFrame, 
                        freq : int=5):
    df2 = _equal_index(df, freq) 
    condition = _remove_nulls(df2, freq)
    df3 = _interpolate_clean(df2, condition)
    
    return df3

def _equal_index(df : pd.DataFrame, 
                 freq : int=5):
    freq_str = str(freq) + 'min'
    df_ = df.set_index('UTC_datetime')
    start_date = df_.index.min()
    rounded_date = start_date - timedelta(minutes=start_date.minute%freq, 
                                          seconds=start_date.second)
    
    df_reindexed = df_.reindex(pd.date_range(start = rounded_date,
                                             end = df_.index.max(),
                                             freq = freq_str))

    df_full = pd.concat([df_, df_reindexed], axis = 0)  
    df_full.sort_index(axis = 0, inplace =True)
    df_full.reset_index(inplace = True)
    df_full.rename(columns = {'index': 'UTC_datetime'}, inplace =True)
    df_full.drop_duplicates(inplace=True)
    return df_full

 
def _remove_nulls(df, freq = 5):
    # Get rows where too much columns are null
    threshold = int(60/freq)
    n_nulls = df['ID'].isna().rolling(threshold, center=False).sum()
    # shift -threshold + 2 pq threshold solo borraria el último valor con dato
    # +1 para salvar la última fecha con dato
    # +2 para salvar la última fecha generada (valor nulo)
    forward_condition = (n_nulls.shift(-threshold+2) != threshold)
    # +1 para salvar la ultima fecha sin dato
    backward_condition = (n_nulls.shift(1) != threshold)
    condition1 = forward_condition & backward_condition
    
    # Get rows of the new indexes
    c1 = df.UTC_datetime.dt.minute%freq == 0 # Multiplos de freq
    c2 = df.UTC_datetime.dt.second == 0 # Segundos = 0
    condition2 = c1 & c2

    condition = condition1 & condition2
    return condition

def _interpolate_clean(df, condition):
    # Same datetime column
    dates = df.UTC_datetime
    dates = dates[condition]
    # Drop datetime columns
    df_ = df.drop(labels = ['UTC_datetime', 'UTC_date', 'UTC_time'], axis = 1)

    df_ = df_.interpolate(method ='linear', limit_direction ='forward')
    df_ = df_[condition]
    df_['UTC_datetime'] = dates
    df_['UTC_date'] = dates.dt.date
    df_['UTC_time'] = dates.dt.time
        
    return df_



def get_same_data(df, lista_nombres):
    
    df2 = df[df['name'].isin(lista_nombres)]
    dg = pd.DataFrame(df2.groupby('UTC_datetime')['name'].nunique())
    dg.rename(columns={'name': 'n_data'}, inplace = True)
    dg.reset_index(inplace =True)
    df2 = df2.merge(dg, on='UTC_datetime', how = 'left')
    df3 = df2[df2['n_data']==len(lista_nombres)]
    
    return df3
# =============================================================================
# CÁLCULOS
# =============================================================================

def deg_km(v):
    dif = abs(v)
    min_dif = np.minimum(dif, 360-dif)
    rad_dif = 2*np.pi*min_dif/360
    km_dif = 6370*rad_dif
    return km_dif  

def calculate_distance(df1, df2):
    lat_arr = df1.Latitude - df2.Latitude
    d_lat = deg_km(lat_arr)
    
    long_arr = df1.Longitude - df2.Longitude
    d_long = deg_km(long_arr)
    
    d_alt = (df1.Altitude_m - df2.Altitude_m)/1000
    
    distancias = pd.DataFrame()
    distancias['3D'] = np.sqrt(d_lat**2 + d_long**2 + d_alt**2)
    distancias['2D'] = np.sqrt(d_lat**2 + d_long**2) 
    distancias['altura'] = d_alt
    
    return distancias

def calculate_distance_intra_df(df, 
                                col_latitude1, col_latitude2,
                                col_longitude1, col_longitude2,
                                col_altitude1, col_altitude2):
    df = df.copy()
    lat_arr = df[col_latitude1] - df[col_latitude2]
    d_lat = deg_km(lat_arr)
    
    long_arr = df[col_longitude1] - df[col_longitude2]
    d_long = deg_km(long_arr)
    
    d_alt = abs(df[col_altitude1] - df[col_altitude2])/1000
    

    df['distance_3D'] = np.sqrt(d_lat**2 + d_long**2 + d_alt**2)
    df['distance_2D'] = np.sqrt(d_lat**2 + d_long**2) 
    df['distance_height'] = d_alt
    
    return df

def time_groupby(df, freq):
    """
    Realiza un agrupamiento temporal y calcula métricas relacionadas con el tiempo de vuelo para un DataFrame dado.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame con los datos de vuelo.

    freq : str
        La frecuencia para el agrupamiento temporal. Puede ser 'daily' o 'hourly'.

    Returns:
    --------
    pandas.DataFrame
        Un DataFrame resultante con el agrupamiento y las métricas calculadas.

    Notas:
    ------
    Esta función realiza un agrupamiento temporal según la frecuencia especificada (diaria o por hora) y calcula
    algunas métricas relacionadas con el tiempo de vuelo, incluyendo la altitud máxima, el tiempo de vuelo total,
    la distancia 2D recorrida y la distancia vertical recorrida.

    Los parámetros para el agrupamiento y cálculo de métricas están definidos en el diccionario 'params', que contiene
    las variables por las que se agrupará el DataFrame y el número máximo de segundos permitidos en el tiempo de vuelo
    según la frecuencia.

    La columna 'flying_stituation' del DataFrame debe contener la información sobre si el pajaro está 'flying' o 'landed'.
    """
    params = {'daily': {'groupby_variables': ['UTC_date'],
                        'max_seconds': 3600*24},
              'hourly': {'groupby_variables': ['UTC_date', 'hour'],
                         'max_seconds': 3600}
              }

    groupby_variables = params[freq]['groupby_variables']
    dg_fly = df.groupby(groupby_variables)\
               .agg(mean_altitude=('Altitude_m', 'mean'),
                    mean_flying_time=('time_step_s', 'mean'),
                    mean_distance_2D=('distance_2D', 'mean'),
                    mean_speed=('speed_km_h', 'mean'),
                    count=('speed_km_h', 'count'))
                       
    fly_variables = list(dg_fly.columns)
    dg_fly = dg_fly.reset_index()
    dg_fly.UTC_date = dg_fly.UTC_date.astype(str) 
    # max_flying_time = params[freq]['max_seconds']
    # dg_fly = dg_fly.assign(flying_time=np.where(dg_fly['flying_time']>max_flying_time, 
    #                                             max_flying_time, 
    #                                             dg_fly['flying_time']))
    return dg_fly, fly_variables

