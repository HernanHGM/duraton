import gradio as gr
from joblib import load
import numpy as np

def scaler_model_selector():
    path = "E:\\duraton\\sexaje\\_dev\\Aguila real\\model_scaler"
    file_model = path + '\\classifier.joblib'
    file_scaler = path + '\\scaler.joblib'
    model = load(file_model) 
    scaler = load(file_scaler)
    
    return model, scaler


def calculate_means(L_izda, L_dcha):
    """
    Calcula la media entre "izda L" y "dcha L"
    """
    L_media = np.nanmean([L_izda, L_dcha])
    return L_media 

def process_inputs(measured_values):
    '''
    En la aplicación se introducen estos datos:
    'peso'
    'L izda'
    'L dcha'
    
    El clasificador recibe: 
    L_media
    peso
    
    Parameters
    ----------
    measured_values : iterable
        valores introducidos por la aplicacion.

    Returns
    -------
    processed_data : numpy array
        array con los datos procesados y con forma (1,2) 
        para ser leidos por el clasificador

    '''
    peso, L_izda, L_dcha = measured_values
    L_media = calculate_means(L_izda, L_dcha)
    
    processed_data = [L_media, peso]
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

title = "Clasificador del sexo de águila real"
description = """
Esta aplicación se ha creado para clasificar el sexo de las águilas reales completamente desarrolladas. 
El modelo de clasificación se ha entrenado con 31 hembras y 34 machos consiguiendo una precisión del 100%
"""
demo = gr.Interface(
    fn=complete_classification,
    inputs=[gr.Number(label = 'Peso (g)'),
            gr.Number(label = 'L izda (mm)'),
            gr.Number(label = 'L dcha (mm)')],
    outputs="text",
    title=title,
    description=description
)
demo.launch()
