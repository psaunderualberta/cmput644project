import os
import pickle
from src.heuristic.simplifier import Simplifier
from glob import glob
import numpy as np
import pandas as pd
from functools import reduce
from sympy import *


def main():
    file_location = os.path.dirname(os.path.realpath(__file__))
    table_files = glob(
        os.path.join(
            file_location,
            "artifacts",
            "**",
            "*.pkl",
        )
    )

    tables = []
    for table_file in table_files:
        with open(table_file, "rb") as f:
            tables.append(pickle.load(f))

    data = list(map(lambda table: table.get_stored_data(strip_nan=True), tables))
    expressions = list(map(lambda d: d[0], data))
    fitnesses = list(map(lambda d: d[1], data))
    origins = []
    for i, arr in enumerate(expressions):
        origins.extend([i] * len(arr))

    simplifier = Simplifier(
        np.ravel(expressions), fitnesses=np.ravel(fitnesses), origins=origins
    )

    print("TOTAL # OF HEURISTICS:", len(simplifier.df))

    print("\n" + "-" * 20 + " NON-SIMPLIFIED " + "-" * 20)
    print("TOTAL # OF UNIQUE HEURISTICS:", len(np.unique(simplifier.df["expressions"])))
    print("HEURISTICS IN COMMON B/W TABLES:", len(reduce(np.intersect1d, expressions)))

    equivalencies = simplifier.get_equivalencies()
    print("\n" + "-" * 20 + " SIMPLIFIED " + "-" * 20)
    print("TOTAL # OF UNIQUE HEURISTICS:", len(equivalencies))
    print("EQUIVALENCIES:")
    for key, values in equivalencies.items():
        if len(values):
            exprs = [str(key)] + [str(v) for v in values]
            print(f"\t" + " == ".join(exprs))

    df = simplifier.get_df()
    df = df.sort_values(by="fitnesses", ascending=False)
    df["latex"] = df["simplified"].apply(latex)

    fname = os.path.join("logs", "synthesized.csv")
    df[["expressions", "latex", "fitnesses", "origins"]].to_csv(fname, sep="$", index=True)


if __name__ == "__main__":
    main()
