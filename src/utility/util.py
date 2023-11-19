import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utility.constants import NORMALIZED_COLUMN_NAMES_MAPPING as mapping
import dask.dataframe as dd
import dask.array as da

def load_data(files, dask=False):
    df = []

    if not dask:
        for file in tqdm(files):
            df.append(pd.read_parquet(file))
        df = pd.concat(df, axis=0)
    else:
        df = dd.read_parquet(files)

    df.columns = list(map(lambda col: mapping[col] if col in mapping else col, df.columns))
    return df

def operators(op, dask=False):
    if dask:
        ops = {
            "plus": ("+", lambda x, y: x + y),
            "sub": ("-", lambda x, y: x - y),
            "mul": ("*", lambda x, y: x * y),
            "div": ("/", lambda x, y: x / y),
            "max": ("max", lambda x, y: da.maximum(x, y)),
            "min": ("min", lambda x, y: da.minimum(x, y)),
            "abs": ("abs", lambda x: da.abs(x)),
            "neg": ("neg", lambda x: -x),
            "sqrt": ("sqrt", lambda x: da.sign(x) * da.sqrt(da.abs(x))),
            "sqr": ("sqr", lambda x: x**2),
        }
    else:
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
