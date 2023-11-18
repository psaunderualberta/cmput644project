import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utility.constants import NORMALIZED_COLUMN_NAMES_MAPPING as mapping

def load_data(files):
    dfs = []
    for f in tqdm(files):
        dfs.append(pd.read_csv(f))

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    combined.columns = list(map(lambda col: mapping[col] if col in mapping else col, combined.columns))
    return combined


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
