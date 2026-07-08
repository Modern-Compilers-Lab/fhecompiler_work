import numpy as np

def generate_pref_list(n_points: int):
    """
    Generates a list of preference vectors.
    Example n_points=2: [[1.0, 0.0], [0.0, 1.0]]
    Example n_points=3: [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    """
    if n_points < 2:
        return [[0.5, 0.5]]
    
    prefs = []
    for i in range(n_points):
        # Linearly interpolate between 0 and 1
        w_keys = round(i / (n_points - 1), 1)
        w_ops = round(1.0 - w_keys, 1)
        prefs.append([w_ops, w_keys])
    return prefs[:-1]