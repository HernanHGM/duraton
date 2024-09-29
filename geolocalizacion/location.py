from flask import Flask
import requests
from bs4 import BeautifulSoup
import pandas as pd

app = Flask(__name__)
@app.route('/api/_get_municipio_info')
def _get_location_info(municipio):
    '''
    Se descarga de la url indicada  en las siguientes líneas, las coordenadas
    geográficas y la elevacion del terreno a la que se encuentra.
    
    Parameters
    ----------
    municipio : str
        Nombre del municipio cuya informacion se va a descargar.

    Returns
    -------
    municipio_info: dict
        Diccionario con el nombre latitud, longitu y elevacion del municipio.

    '''
    municipio_url = '-'.join(municipio.lower().split(' '))
    url = f'https://www.ayuntamiento-espana.es/ayuntamiento-{municipio_url}.html'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Comprobar si hay errores en la solicitud

        soup = BeautifulSoup(response.content, 'html.parser')

        altitud_th = soup.find('th', text=f'Altitud del municipio de {municipio}')
        # Buscar el valor de Altitud en la siguiente etiqueta <td>
        altitude_value = altitud_th.find_next('td').text.strip().split()[0]
        
        # Buscar la etiqueta <strong> que contiene 'Latitud:'
        latitud_strong = soup.find('strong', text='Latitud:')
        # Buscar el valor de Latitud en la siguiente etiqueta <span> con clase 'latitude'
        latitude_value = latitud_strong.find_next('span', class_='latitude').text.strip()
        
        # Buscar la etiqueta <strong> que contiene 'Latitud:'
        longitud_strong = soup.find('strong', text='Longitud:')
        # Buscar el valor de Latitud en la siguiente etiqueta <span> con clase 'latitude'
        longitud_value = longitud_strong.find_next('span', class_='longitude').text.strip()


        # Crear un diccionario con la información extraída
        municipio_info = {
            'Location': municipio,
            'Latitude': latitude_value,
            'Longitude': longitud_value,
            'Altitude': altitude_value
        }
        print(municipio, ' okey')
        return municipio_info  # Devolver la información en formato JSON

    except requests.exceptions.RequestException as e:
        print(f'Municipio_info for {municipio} was not created')
        print(f'Error en la solicitud web: {e}', 500)


def download_location_info(location_list: list, 
                           save: bool=False, 
                           app: Flask=app):
    '''
    Llama a la funcion _get_location_info para descargar la informacion de 
    la lista de municpios indicados.
    Acepta la mayoría de pueblos y ciudades pero solo de España 
    escrito siguiendo las normas ortográficas: mayúsculas y tildes
    almeria-->MAL
    Almeria-->MAL
    almería-->MAL
    Almería-->Bien
    si solo se pasa un string, se convierte en lista y funciona igual.

    Parameters
    ----------
    location_list : list
        Lista de municipios cuya informacion se quiere.
        Ejemplo
        location_list = ['Romangordo', 'Deleitosa', 'Torrecillas de la Tiesa', 
                          'Herguijuela', 'Conquista de la Sierra', 'Zorita', 
                          'Alcollarín']
    save : bool, optional
        save=True guarda el dataFrame
    app : Flask, optional
        La app que genera la llamada a la url de la que se descargan datos.

    Returns
    -------
    df_coordinates : pd.DataFrame
        dataframe con la información de cada municipio.
    '''
    if isinstance(location_list, str):
        location_list = [location_list]
    with app.app_context():
        info_list = [_get_location_info(location) for location in location_list]
    
    # Clean in case of error only properly created jsons are used
    info_list = [item for item in info_list if isinstance(item, dict)]
    df_coordinates = pd.DataFrame(info_list)
    
    if save==True:
        path = 'E:/duraton/geolocalizacion/_data/weather/all_locations/coordinates.csv'
        df_coordinates.to_csv(path, index=False, encoding="ISO-8859-1")
    return df_coordinates

