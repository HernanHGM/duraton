import gradio as gr

def promediado(L_izda, DV_izda, L_dcha, DV_dcha):
    L = (L_izda+L_dcha)/2
    DV = (DV_izda+DV_dcha)/2
    return L, DV

def clasificador(L, DV):
    
    import numpy as np
    from joblib import load
    
    file_model = 'E:\\trabajo_pajaros\\marcajes\\model.pkl'
    file_scaler = 'E:\\trabajo_pajaros\\marcajes\\scaler.pkl'
    
    model = load(file_model) 
    scaler = load(file_scaler)
    
    data = np.array([L, DV]).reshape(1, -1)
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    
    sexo = pred[0]
    
    return sexo


def clasificador_completo(L_izda, DV_izda, L_dcha, DV_dcha):
    
    L, DV = promediado(L_izda, DV_izda, L_dcha, DV_dcha)
    sexo = clasificador(L, DV)
    
    return sexo

x = [13.8, 17.3, 13.8, 17.5]
sexo = clasificador_completo(x[0],x[1],x[2],x[3])

title = "Clasificador del sexo de buitre negro"
description = """
Esta aplicación se ha creado para clasificar el sexo de los pollos de buitres negros. 
Para ello basta con tomar las medidas lateral y dorso-ventral de ambos tarsos del individuo.
Los datos a introducir son los siguientes \n
    - L_izda: medida lateral del tarso de la pata izquierda \n
    - DV_izda: medida dorso-ventral del tarso de la pata izquierda \n
    - L_dcha: medida lateral del tarso de la pata derecha \n
    - DV_izda: medida dorso-ventral del tarso de la pata derecha \n
"""

demo = gr.Interface(
    fn=clasificador_completo,
    inputs=["number", "number", "number", "number"],
    outputs="text",
    title=title,
    description=description
)
demo.launch(share = True)
demo