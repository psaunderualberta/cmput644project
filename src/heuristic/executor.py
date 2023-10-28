from src.heuristic.expressions import Binary, Unary, Terminal, Number
import numpy as np
import pandas as pd

def execute(heuristic, data):
    if isinstance(heuristic, Binary):
        left = execute(heuristic.left, data)
        right = execute(heuristic.right, data)
        return heuristic.fun(left, right)

    elif isinstance(heuristic, Unary):
        return heuristic.fun(execute(heuristic.right, data))

    elif isinstance(heuristic, Terminal):
        return data[heuristic.data].to_numpy()
    
    elif isinstance(heuristic, Number):
        return np.ones(data.shape[0]) * heuristic.value