# %% DESCRIPTION
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:19:52 2023

@author: Hernán García Mayoral

Module of functions dedicated to enrich a geopositions dataframe with the 
distance form one bird to the rest that appear on that dataframe
Is important that the primary key is composed of the name of the bird and 
the datetime of the geoposition.
the datetime can be together at a single column or divided on several variables
"""
# %% LIBRARIES IMPORT
import pandas as pd
import numpy as np
from typing import List


import sys

path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

# OWN LIBRARIES
import geolocalizacion.data_processing as dp
# %% FUCNTIONS
def add_distances_interbirds(df: pd.DataFrame, 
                             datetime_cols: List[str]=['UTC_datetime'])->pd.DataFrame:
    '''
    Enrich the received DataFrame with the distances to all the birds 
    appearing inside it.
    The process divide the original dataframe in as many dataframes as birds 
    appear, then it enrich each sub dataframe with the distances to the rest
    of the birds. 
    In the end, each subdataframe contains the same columns, so they are 
    concatenated. The rows of the original anf final dataframe are the same
    but the columns in the end must be n_end = n_original + number_birds
    Distance from one bird to itself must be 0 at every time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing all the data.
    datetime_cols : List[str], optional
        list of columns that together compose the datetime. 
        The default is ['UTC_datetime'].

    Returns
    -------
    df_joined : pd.DataFrame
        original dataframe but enriched with the distances to other birds.

    '''
    df_enriched_list = []
    
    for name1 in df.name.unique():
        df_enriched = df[df.name == name1]
        for name2 in df.name.unique():
            print('Enriching:', name1, name2)
            df1, df2 =  _select_birds_and_columns(df, name1, name2, datetime_cols)  
            df1_filtered, df2_filtered = _get_same_datetimes(df1, df2, datetime_cols)
            distance = _calculate_distance(df1_filtered, df2_filtered)
            df1_distances = _get_distances_to_bird(distance, df1_filtered, name2, datetime_cols)
            df_enriched = _add_distances_partial_df(df_enriched, df1_distances, datetime_cols)
            
        df_enriched_list.append(df_enriched)  
    
    df_joined = pd.concat(df_enriched_list, axis=0)
    return df_joined

def _select_birds_and_columns(df: pd.DataFrame,
                             name1: str,
                             name2: str,
                             datetime_cols: List[str]=['UTC_datetime']):
    '''
    Selects birds (rows) and useful columns for the process.
    '''
    columns = ['name', 'Latitude', 'Longitude', 'Altitude_m'] + datetime_cols
    df1 = df.loc[df.name==name1, columns]
    df2 = df.loc[df.name==name2, columns]
    return df1, df2

def _get_same_datetimes(df1: pd.DataFrame, 
                       df2: pd.DataFrame, 
                       datetime_cols: List[str]=['UTC_datetime']) -> pd.DataFrame:
    '''
    Selects the coincident rows between two dataframes regarding the datetime.
    Parameters
    ----------
    df1 : pd.DataFrame
    df2 : pd.DataFrame
    datetime_col : List[str]
        list of column whose values represent datetimes on both dataframes, 
        and we are looking to be equal.

    Returns
    -------
    both dataframes with only the coincident datetime values.

    '''
    
    date_coincidences = pd.merge(df1[datetime_cols], df2[datetime_cols],
                                on=datetime_cols,
                                how='inner')
    
    df1_coincident = pd.merge(df1, date_coincidences,
                              on=datetime_cols,
                              how='inner')
    
    df2_coincident = pd.merge(df2, date_coincidences,
                              on=datetime_cols,
                              how='inner')
    return df1_coincident, df2_coincident

def _calculate_distance(df1: pd.DataFrame,
                        df2: pd.DataFrame)-> pd.DataFrame:
    '''
    calculates 2D, 3D and height difference between two birds for each 
    coincident datetime

    Parameters
    ----------
    df1 : pd.DataFrame
        all geopositions of bird 1.
    df2 : pd.DataFrame
        all geopositions of bird 2.
    Returns
    -------
    distances : pd.DataFrame
        distances between the coincident rows of two birds.

    '''
    lat_arr = df1.Latitude - df2.Latitude
    d_lat = dp.deg_km(lat_arr)
    
    long_arr = df1.Longitude - df2.Longitude
    d_long = dp.deg_km(long_arr)
    
    d_alt = (df1.Altitude_m - df2.Altitude_m)/1000
    
    df_distances = pd.DataFrame()
    df_distances['3D'] = np.sqrt(d_lat**2 + d_long**2 + d_alt**2)
    df_distances['2D'] = np.sqrt(d_lat**2 + d_long**2) 
    df_distances['altitude_difference'] = d_alt
    
    return df_distances

def  _get_distances_to_bird(distance: pd.DataFrame,
                            df1_coincident: pd.DataFrame,
                            name2: str, 
                            datetime_cols: List[str]=['UTC_datetime'])-> pd.DataFrame:
    '''
    Creates the minimum information dataframe containg the PK [name, datetime_cols]
    of bird 1 and the distances to bird 2 to each row.    

    Parameters
    ----------
    distance : pd.DataFrame
        dataframe containig the 2D, 3D and altitud distances between two birds.
    df1_coincident : pd.DataFrame
        df of the bird 1, only with the rows that exist for both bird 1 and bird 2.
    name2 : str
        name of the bird 2.
    datetime_cols : List[str], optional
        list of columns that define the datetime of a certain data.
        The default is ['UTC_datetime'].

    Returns
    -------
    df1_distances : pd.DataFrame
        minimum information dataframe containg the PK [name, datetime_cols]
        of bird 1 and the distances to bird 2 to each row.

    '''
    join_columns = ['name'] + datetime_cols
    df1_distances = pd.merge(df1_coincident[join_columns], 
                     distance['2D'], 
                     left_index=True, 
                     right_index=True)\
            .rename(columns = {'2D': f'km_{name2}'})

    return df1_distances

def _add_distances_partial_df(df_enriched: pd.DataFrame, 
                              df1_distances: pd.DataFrame, 
                              datetime_cols: List[str]=['UTC_datetime']) -> pd.DataFrame:
    '''
    Enrich a dataframe that contain data of bird1 with a column 
    containing the distances to bird2 called 'km_{namebird2}'
    this function is used itertively to add to the same bird the distances 
    to several birds
    
    Parameters
    ----------
    df_enriched : pd.DataFrame
        df with the data of bird1 that will be enriched with a column 
        containing the distances to bird2.
    df1_distances : pd.DataFrame
        dataframe with the distances between two birds.
        bird 1 is set on the column 'name'
        bird 2 is set on column 'km_{namebird2}'
    datetime_cols : List[str], optional
        list of columns that define the datetime of a certain data.
        The default is ['UTC_datetime'].

    Returns
    -------
    df_enriched : pd.DataFrame
        original dataframe with data of one bird enriched with the distances 
        to another bird.

    '''
    join_columns = ['name'] + datetime_cols
    df_enriched = pd.merge(df_enriched,
                           df1_distances,
                           on=join_columns, 
                           how = 'left')
    return df_enriched



