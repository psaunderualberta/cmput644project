import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(files):
    dfs = []
    for f in tqdm(files):
        dfs.append(pd.read_csv(f))
    
    return pd.concat(dfs, axis=0, ignore_index=True)

def operators(op):
    ops = {
        "plus": ("+", lambda x, y: x + y),
        "sub": ("-", lambda x, y: x - y),
        "mul": ("*", lambda x, y: x * y),
        "div": ("/", lambda x, y: x / y),
        "max": ("max", lambda x, y: np.max([x, y], axis=0)),
        "min": ("min", lambda x, y: np.min([x, y], axis=0)),
        "abs": ("abs", lambda x: np.abs(x)),
        "neg": ("neg", lambda x: -x),
        "sqrt": ("sqrt", lambda x: np.sign(x) * np.sqrt(np.abs(x))),
        "sqr": ("sqr", lambda x: x**2),
    }

    return ops[op]
