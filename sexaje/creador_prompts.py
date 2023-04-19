# =============================================================================
# 1.Lee funcion
# =============================================================================
def leer_archivo_py(archivo):
    """
    Lee un archivo .py y devuelve su contenido como una cadena de texto.

    Args:
        archivo (str): Ruta del archivo .py a leer.

    Returns:
        str: Contenido del archivo .py como una cadena de texto.
    """
    try:
        with open(archivo, 'r') as f:
            contenido = f.read()
        return contenido
    except FileNotFoundError:
        print(f"El archivo '{archivo}' no se encontró.")
        return None
    except Exception as e:
        print(f"Error al leer el archivo '{archivo}': {e}")
        return None

path = "C:\\duraton\\sexaje\\clase_limpieza.py"
a = leer_archivo_py(path)


# =============================================================================
# 2.Documenta funcion
# =============================================================================
# Pedir a chat gpt documentar y comentar funcion

# =============================================================================
# 3.generar prompt a partir de documentacion
# =============================================================================
meta_prompt = '''
Genera la variable 'prompt' que contenga con la siguiente estructura, la información mínima viable para replicar la función calculate_correlation.
prompt = """
    Nombre:
    Documentación:
    Parámetros:
    Acciones:
    Returns:
"""
'''

# =============================================================================
# 4.Genera fucion desde prompt
# =============================================================================
# genera una método a partir de la imformación almacenada en la variable prompt:
