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

def load_preprocess_data(folder_path: str, 
                          filenames: List[str], 
                          origin: str='movebank',
                          birds_info_path: str="E:\\duraton\\geolocalizacion\\_data\\fly\\raw\\birds_info.xlsx",
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
        
    origin : str
        Indicates the origin of the data. It accepts the values:
            'ornitela' or 'movebank'.
            
    birds_info_path : str
        excel path containg a table with the name and specie corresponding to each ID
        
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
    full_filepaths = (folder_path + '\\' + filename for filename in filenames) 
    
    li = []
    for filepath in full_filepaths:
        print(filepath)
        df = read_adjust_files(filepath, origin)
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
            df = enriching(df, birds_info_path)
        li.append(df)
    
    df = pd.concat(li, axis=0, ignore_index=True)
    return df

def read_adjust_files(filepath: str,
                      origin: str)-> pd.DataFrame:
    '''
    Due to the different types of data structures based on its origin: Ornitella
    or Movebank, this function adjust the columns of movebank structure to the 
    Ornitella's structure

    Parameters
    ----------
    filepath : str
        path where the csv is stored.
    origin : str
        Indicates the origin of the data. It accepts the values:
            'ornitela' or 'movebank'.

    Returns
    -------
    df : pd.DataFrame
        dataframe normalized to the same structure.

    '''

    
    if origin=='movebank':
        df = pd.read_csv(filepath,
                         parse_dates = ["timestamp"], 
                         index_col=False,
                         encoding="utf-8")
        columns_mapping = {"tag-local-identifier":"ID",
                          "timestamp": "UTC_datetime", 
                          "gps:satellite-count": "satcount", 
                          "location-lat": "Latitude", 
                          "location-long": "Longitude", 
                          "height-above-msl": "Altitude_m", 
                          "ground-speed": "speed_km_h", 
                          "mag:magnetic-field-raw-x": "mag_x", 
                          "mag:magnetic-field-raw-y": "mag_y", 
                          "mag:magnetic-field-raw-z": "mag_z", 
                          "acceleration-raw-x": "acc_x", 
                          "acceleration-raw-y": "acc_y", 
                          "acceleration-raw-z": "acc_z"}        
    
        # Usar el diccionario de mapeo para renombrar las columnas en tu DataFrame
        df.rename(columns=columns_mapping, inplace=True)
        df['UTC_date'] = df.UTC_datetime.dt.date
        df['UTC_time'] = df.UTC_datetime.dt.time
        df['speed_km_h'] = df['speed_km_h']*3.6 #Movebank uses m/s
        useful_variables = list(columns_mapping.values()) + ['UTC_date', 'UTC_time']
        df = df[useful_variables]
    
    elif origin=='ornitela':
        df = pd.read_csv(filepath,
                         parse_dates = ['UTC_datetime'], 
                         index_col=False,
                         encoding="ISO-8859-1")
    else:
        raise ValueError(f'origin accepts "ornitela" or "movebank" but "{origin}" was received')
    

    return df

def _clean_unnamed(df):
    columns = df.columns.to_list()
    unnamed_cols = [col for col in columns if 'Unnamed' in col]
    df = df.drop(labels=unnamed_cols, axis=1)
    return df

def enriching(df : pd.DataFrame, 
               birds_info_path: str):
    # Añadimos la columna de hora
    df = df.assign(hour=df.UTC_datetime.dt.strftime('%H'))
    df['hour']= pd.to_datetime(df['hour'], format='%H').dt.time.astype(str)
    df['UTC_datetime_pre'] = df['UTC_datetime'].shift(1)
    df['time_step_s'] = (df['UTC_datetime']-df['UTC_datetime_pre']).dt.seconds
    df['month_name'] = pd.to_datetime(df['UTC_date'], format="%Y/%m/%d")\
                         .dt.strftime('%B')
    df['month_number'] = pd.to_datetime(df['UTC_date'], format="%Y/%m/%d")\
                         .dt.strftime('%m')
    df['week_number'] = pd.to_datetime(df['UTC_date'], format="%Y/%m/%d")\
                          .dt.strftime('%W')
                          
     
    # corremos una posicion los datos GPS, _lag = dato de la posición previa
    df['Latitude_lag'] = df['Latitude'].shift(1)
    df['Longitude_lag'] = df['Longitude'].shift(1)
    df['Altitude_m_lag'] = df['Altitude_m'].shift(1)
    # Calculamos distancias recorridas usando datos de posicion
    # Dato actual - dato posición previa
    df = calculate_distance_intra_df(df,
                                     'Latitude', 'Latitude_lag',
                                     'Longitude', 'Longitude_lag',
                                     'Altitude_m', 'Altitude_m_lag')   

    df['acc'] = np.sqrt(df.acc_x**2 + df.acc_y**2 + df.acc_z**2)
    df['mag'] = np.sqrt(df.mag_x**2 + df.mag_y**2 + df.mag_z**2)
    
    # Add name and specie
    df_birds_info = pd.read_excel(birds_info_path)
    df = df.merge(df_birds_info, how = 'left', on = 'ID')

    df['breeding_period'] = df[['UTC_datetime','specie']].apply(_categorize_breeding_period, axis = 1)

    return df

def _categorize_breeding_period(row : pd.DataFrame):
    '''
    There are three breeding periods and a no breeding period, 4 categories
    Based on the specie the dates change

    Parameters
    ----------
    row : pd.DataFrame
        UTC_datetime column contains the date.
        specie column contains the specie.

    Returns
    -------
    period_value : str
        incubation, breeding, chick dependency or no breeding period.

    '''
    if row['specie'] == 'Aquila fasciata':
        if (row['UTC_datetime'].month == 2 and row['UTC_datetime'].day >= 15) or \
           (row['UTC_datetime'].month == 3 and row['UTC_datetime'].day <= 31):
            period_value = 'incubation'
            
        elif (row['UTC_datetime'].month == 4 and row['UTC_datetime'].day >= 1) or \
           (row['UTC_datetime'].month == 5 and row['UTC_datetime'].day <= 31):
            period_value = 'breeding'

        elif (row['UTC_datetime'].month == 6 and row['UTC_datetime'].day >= 1) or \
           (row['UTC_datetime'].month == 7 and row['UTC_datetime'].day <= 31):
            period_value = 'chick dependency'
        else:
            period_value = 'no breeding period'
            
    elif (row['specie'] == 'Aquila adalberti') or \
       (row['specie'] == 'Aquila chrysaetos'):
        if (row['UTC_datetime'].month == 3 and row['UTC_datetime'].day >= 15) or \
           (row['UTC_datetime'].month == 4 and row['UTC_datetime'].day <= 31):
               #15 March 30 abril
               period_value = 'incubation'
            
        elif (row['UTC_datetime'].month == 5 and row['UTC_datetime'].day >= 1) or \
             (row['UTC_datetime'].month == 6 and row['UTC_datetime'].day <= 31) or \
             (row['UTC_datetime'].month == 7 and row['UTC_datetime'].day <= 10):
                 #1 May 10 July
                 period_value = 'breeding'

        elif (row['UTC_datetime'].month == 7 and row['UTC_datetime'].day >= 11) or \
             (row['UTC_datetime'].month == 8 and row['UTC_datetime'].day <= 31):
                 #11 July 31 August
                 period_value = 'chick dependency'
        else:
            period_value = 'no breeding period'
    else:
        raise ValueError(f"specie accepts Aquila adalberti, Aquila chrysaetos or Aquila fasciata but {row['specie']} was received")
            
    return period_value




def create_bird_info_table(folder_path: str, save_path: str):
    '''
    Creates quiqly a table info from all the birds in .csv format saved in 
    a certain folder. The data stored is the ID, name of the bird and its specie 

    Parameters
    ----------
    folder_path : str
        path of the folder where all the .csv are stores
    save_path : str
        path wher the excel containing the birds information will be saved.

    Returns
    -------
    None.

    '''
    filenames = find_csv_filenames(folder_path)
    
    full_filepaths = (folder_path + '\\' + filename for filename in filenames) 
    
    li = []
    for filepath in full_filepaths:
        print(filepath)
        df = pd.read_csv(filepath, 
                         index_col=False,
                         encoding="utf-8")
        specie = df["individual-taxon-canonical-name"].unique()[0]
        ID = df["tag-local-identifier"].unique()
        if len(ID)>1:
            print(ID) #This never should be printed, just on bird by file
        ID = ID[0]
        name = df["individual-local-identifier"].unique()[0]
        li.append((specie, ID, name))
    
    
    df_info = pd.DataFrame(li, columns =['specie', 'ID', 'name'])
    df_info.to_excel(save_path, index=False, header=True)
 
# UNUSED. A table containig containig explicitly the information has been created
def extract_info(nombre_archivo):
    nombre_limpio = nombre_archivo.replace('.csv', '')
    partes = nombre_limpio.split('_')
    
    conversor_especie={'Aquada': 'Aquila adalberti',
                        'Aquchr': 'Aquila chrysaetos',
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
    """
    Remove rows from a DataFrame based on specified conditions related to null values and time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame from which rows are to be removed.
    freq : int, optional
        The frequency for time conditions, by default 5.
    
    Returns:
    --------
    pandas.Series
        A boolean series indicating the rows to be removed (True) or 
        kept (False).
    
    Notes:
    ------
    - The function combines conditions related to the number of null values 
    in a rolling window and specific time conditions.
    - Rows satisfying the combined conditions are marked as True, 
    and the resulting Series can be used for filtering the DataFrame.
    """
    # Get rows where too much columns are null
    # When the period between two data is over one hour, the intermedium 
    # interpolated data will be erased. Is too much time to be a realistic interpolation
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


def calculate_distance_intra_df(df, 
                                col_latitude1, col_latitude2,
                                col_longitude1, col_longitude2,
                                col_altitude1, col_altitude2):
    df = df.copy()
    lat_arr = df[col_latitude1] - df[col_latitude2]
    d_lat = deg_km(lat_arr)
    
    long_arr = df[col_longitude1] - df[col_longitude2]
    d_long = deg_km(long_arr)
    
    d_alt = (df[col_altitude1] - df[col_altitude2])/1000
    

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

