# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:57:45 2023

@author: Hernán_HGM

Download Latitude, longitude and altitude form villages using location.py file
Those coordinates are later used to determinate the closest village to the 
flying bird and merge the fly data with the weather data of that location
"""
# %% 1. IMPORTACION LIBRERÍAS
import pandas as pd

import sys
path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

from geolocalizacion import location
# %% DOWNLOAD COORDINATES

location_list = ['Romangordo', 'Deleitosa', 'Torrecillas de la Tiesa', 
                  'Herguijuela', 'Conquista de la Sierra', 'Zorita', 
                  'Alcollarín', 'Abertura', 'Campo Lugar',
                  'Higuera', 'Casas de Miravete', 'Almaraz', 
                  'Campillo de Deleitosa', 'Aldeacentenera', 'Madroñera', 
                  'Trujillo', 'Garciaz', 'Santa Cruz de la Sierra',
    "Casar de Cáceres",
    "Santiago del Campo",
    "Villar del Rey",
    "Alburquerque",
    "Puebla de Obando",
    "Almoharín",
    "Valdemorales",
    "Montánchez",
    "Arroyomolinos",
    "Casas de Don Antonio",
    "Albalá",
    "Torre de Santa María",
    "Casas de Don Pedro",
    "Talarrubias",
    "La Coronada",
    "Orellana la Vieja",
    "Esparragosa de Lares",
    "Campanario",
    "Quintana de la Serena",
    "Castuera",
    "Benquerencia de la Serena",
    "Valle de la Serena",
    "Higuera de la Serena",
    "Guareña",
    "Manchita",
    "Don Álvaro",
    "Valverde de Mérida",
    "Villagonzalo",
    "La Zarza",
    "Oliva de Mérida",
    "Palomas",
    "Alange",
    "Arroyo de San Serván",
    "Valencia del Ventoso",
    "Fuente de Cantos",
    "Calera de León",
    "Cabeza la Vaca",
    "Montemolín",
    "Monesterio",
    "Santa Cruz del Retamar",
    "Quismondo",
    "Huecas",
    "Rielves",
    "Villamiel de Toledo",
    "Gálvez",
    "Menasalbas",
    "Cuerva",
    "Las Ventas con Peña Aguilera",
    "Sonseca",
    "Orgaz"
]
'''
ArroyoMolinos y la zarza hay que hacerlos a pelo
 porque hay nombres duplicados de pueblos en otras comunidades
El resto de pueblos están bien
'''
df_coordinates = location.download_location_info(location_list)
# %% SAVE
file_path = 'E:/duraton/geolocalizacion/_data/weather/all_locations/coordinates.csv'
df_coordinates.to_csv(file_path, index=False, encoding = "ISO-8859-1")

# %% LOAD
df_ = pd.read_csv(file_path, encoding = "ISO-8859-1")

