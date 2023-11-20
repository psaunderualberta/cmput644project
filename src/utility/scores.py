import dask.array as da
import dask.dataframe as dd
import numpy as np
from .constants import *
import pandas as pd


def get_metrics(actual, pred):

    df = dd.from_pandas(pd.DataFrame(columns=["accuracy", "precision", "recall", "f1"], dtype=np.float64, index=[0]))

    # Compute accuracy, insert into df
    accuracy = da.mean(actual == pred)
    df["accuracy"] = accuracy
    
    # Compute precision score
    tp = da.logical_and(actual == ATTACK_CLASS, pred == ATTACK_CLASS).sum()
    fp = da.logical_and(actual == ATTACK_CLASS, pred == BENIGN_CLASS).sum()
    df["precision"] = tp / (tp + fp)

    # Compute recall score
    fn = da.logical_and(actual == BENIGN_CLASS, pred == ATTACK_CLASS).sum()
    df["recall"] = tp / (tp + fn)

    # Compute f1 score
    df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"])

    return df
