import gradio as gr


def clasificador(measured_value):
    
    import numpy as np
    from joblib import load
    
    file_scaler = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\scaler_longpico.pkl'
    file_model = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Arpia\\model_longpico.pkl'
    
    model = load(file_model) 
    scaler = load(file_scaler)
    
    data = np.array([measured_value]).reshape(1, -1)
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    
    label = ['hembra', 'macho']
    sexo = label[pred[0]]
    
    return sexo


x = 60
sexo = clasificador(x)

title = "Clasificador del sexo de buitre negro"
description = """
Esta aplicación se ha creado para clasificar el sexo de las arpías completamente desarrolladas. 
Para ello basta con medir la longitu del pico.
"""

demo = gr.Interface(
    fn=clasificador,
    inputs=["number"],
    outputs="text",
    title=title,
    description=description
)
demo.launch(share = True)
demo