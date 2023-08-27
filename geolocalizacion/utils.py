
def conditional_selection(array, lim, n, operation='<'):
    """
    Encuentra los índices de valores consecutivos en un array que cumplan una condición.

    Args:
        array (numpy.ndarray): El array de valores.
        lim (float): El valor límite para la condición.
        n (int): El número mínimo de valores consecutivos para considerar.
        operation (str): La operación lógica a aplicar ('>', '<', '==', '!=').

    Returns:
        list: Lista de índices de valores consecutivos que cumplen la condición.
    """
    consecutive_indices = []  # Lista para almacenar los índices consecutivos que cumplen la condición.
    current_group = []  # Lista temporal para mantener el seguimiento de los índices del grupo actual.

    for i, value in enumerate(array):
        if operation == '>' and value > lim:
            current_group.append(i)
        elif operation == '<' and value < lim:
            current_group.append(i)
        elif operation == '==' and value == lim:
            current_group.append(i)
        elif operation == '!=' and value != lim:
            current_group.append(i)
        else:
            if len(current_group) >= n:  # Verificar si el grupo actual cumple el requisito de longitud.
                consecutive_indices.extend(current_group)  # Agregar los índices del grupo a la lista final.
            current_group = []  # Restablecer la lista del grupo actual.
            
    if len(current_group) >= n:  # Agregar los índices del último grupo si cumple el requisito de longitud.
        consecutive_indices.extend(current_group)
        
    return consecutive_indices