import gradio as gr
from joblib import load
import numpy as np

def scaler_model_selector():
    path = "E:\\duraton\\sexaje\\_dev\\Buitre Negro\\model_scaler"
    file_model = path + '\\classifier.joblib'
    file_scaler = path + '\\scaler.joblib'
    model = load(file_model) 
    scaler = load(file_scaler)
    
    return model, scaler


def calculate_means(L_izda, L_dcha, DV_izda, DV_dcha):
    """
    Calcula la media entre "izda L" y "dcha L" 
    Calcula la media entre "izda DV" y "dcha DV"
    """
    L_media = np.nanmean([L_izda, L_dcha])
    DV_media = np.nanmean([DV_izda, DV_dcha])
    return L_media, DV_media

def process_inputs(measured_values):
    '''
    En la aplicación se introducen de esta manera:
    'L izda'
    'L dcha'
    'DV izda'
    'DV dcha'
    'antebrazo'
    'long pico'
    'alto pico'
    'longitud total'
    'longitud cabeza'
    
    Mientras que los datos han de entrar al classificador en este orden:
    long cabeza  
    longitud total  
    L_media  
    alto pico  
    DV_media  
    antebrazo  
    long pico
    

    Parameters
    ----------
    measured_values : iterable
        valores introducidos por la aplicacion.

    Returns
    -------
    processed_data : numpy array
        array con los datos ordenados y con forma (1,7) 
        para ser leidos por el clasificador

    '''
    L_izda, L_dcha, DV_izda, DV_dcha, \
    antebrazo, long_pico, alto_pico,  \
    long_total, long_cabeza= measured_values
    
    L_media, DV_media = calculate_means(L_izda, L_dcha, DV_izda, DV_dcha)
    processed_data = [long_cabeza,
                      long_total,
                      L_media,
                      alto_pico,
                      DV_media,
                      antebrazo,
                      long_pico]
    
    processed_data = np.array([processed_data]).reshape(1, -1)
    return processed_data

def classifier(data, model, scaler):
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    
    label = ['hembra', 'macho']
    sexo = label[pred[0]]
        
    return sexo


def complete_classification(*measured_values):
    model, scaler = scaler_model_selector()
    data = process_inputs(measured_values)
    sexo = classifier(data, model, scaler)
    
    return sexo

title = "Clasificador del sexo de buitre negro"
description = """
Esta aplicación se ha creado para clasificar el sexo de los buitres negros completamente desarrolladas. 
El modelo de clasificación se ha entrenado con 43 machos y 33 hembras condiguiendo una precisión del 92%


"""
demo = gr.Interface(
    fn=complete_classification,
    inputs=[gr.Number(label = 'L izda (mm)'),
            gr.Number(label = 'L dcha (mm)'),
            gr.Number(label = 'DV izda (mm)'),
            gr.Number(label = 'DV dcha (mm)'),
            gr.Number(label = 'antebrazo (cm)'),
            gr.Number(label = 'long pico (mm)'),
            gr.Number(label = 'alto pico (mm)'),
            gr.Number(label = 'longitud total (cm)'),
            gr.Number(label = 'longitud cabeza (mm)')],
    outputs="text",
    title=title,
    description=description
)
demo.launch()
# demo