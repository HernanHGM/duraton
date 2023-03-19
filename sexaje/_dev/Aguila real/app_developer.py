import gradio as gr

def scaler_model_selector(variable):
    from joblib import load
    
    if variable == 'Longitud de pico':
        file_scaler = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\scaler_longpico.pkl'
        file_model = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\model_longpico.pkl'
    elif variable == 'Altura de pico':
        file_scaler = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\scaler_altopico.pkl'
        file_model = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\model_altopico.pkl'  
    elif variable == 'Cola':
        file_scaler = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\scaler_cola.pkl'
        file_model = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\model_cola.pkl'  
    elif variable == 'Peso':
        file_scaler = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\scaler_peso.pkl'
        file_model = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\model_peso.pkl'  
    elif variable == 'Longitud total':
        file_scaler = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\scaler_longitudtotal.pkl'
        file_model = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\model_longitudtotal.pkl'  
 
    model = load(file_model) 
    scaler = load(file_scaler)
    
    return model, scaler

def classifier(measured_value, model, scaler):
    import numpy as np
        
    data = np.array([measured_value]).reshape(1, -1)
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    
    label = ['hembra', 'macho']
    sexo = label[pred[0]]
        
    return sexo


def complete_classification(variable, measured_value):
    model, scaler = scaler_model_selector(variable)
    sexo = classifier(measured_value, model, scaler)
    
    return sexo

variables_opcionales = ['Longitud de pico', 'Altura de pico', 'Cola', 'Peso', 'Longitud total']
title = "Clasificador del sexo de buitre negro"
description = """
Esta aplicación se ha creado para clasificar el sexo de las arpías completamente desarrolladas. 
Para ello basta con medir la longitud del pico en milímetros.

A ver que nos conocemos:
    
    Punto 1: las medidas se meten en MILÍMETROS. Si se mide en centímetros no funciona, tienen que ser MILÍMETROS.

    Punto 2: para los decimales se usan puntos "." o comas ",". Nada más ¿estamos?

"""

demo = gr.Interface(
    fn=complete_classification,
    inputs=[gr.Radio(variables_opcionales),
            gr.Number()],
    outputs="text",
    title=title,
    description=description
)
demo.launch()
# demo