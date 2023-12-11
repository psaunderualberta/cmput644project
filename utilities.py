import os
import sys

import numpy as np
import pandas as pd
from sympy import latex

from src.heuristic.parsing import parse_heuristic
from src.heuristic.simplifier import Simplifier


def top_k_to_latex_table():
    assert (
        len(sys.argv) == 4
    ), "Usage of top_k_to_latex_table: `python utilities.py top_k_to_latex_table <k> <logfile> "
    try:
        k = int(sys.argv[2])
    except ValueError as e:
        print(e.args[0])
        exit()

    fname = sys.argv[3]
    assert os.path.isfile(fname), f"Error: file '{fname}' could not be found."

    df = pd.read_csv(fname, sep="|", engine="python")
    df = df.apply(lambda s: s.strip() if isinstance(s, str) else s)
    df.columns = list(map(str.strip, df.columns))

    # Sort based on fitness value
    df = df.sort_values(by="Fitness", ascending=False).reset_index()

    # Convert to simplified format
    simplifier = Simplifier()
    for i in range(k):
        h = df.loc[i]["Heuristic"]
        parsed_h = parse_heuristic(h)
        fitness = np.round(df.loc[i]["Fitness"], 2)
        simplified = latex(simplifier.symbolic(h, True))
        size = parsed_h.size()
        depth = parsed_h.depth()
        print(f"\t{i+1} & ${simplified}$ & {fitness} & {size} & {depth} \\\\")


def main():
    pass


if __name__ == "__main__":
    utilities = {
        top_k_to_latex_table.__name__: top_k_to_latex_table,
    }
    assert (
        len(sys.argv) > 1
    ), "Error: Must specify a utility to use. Valid utilities are: " + str(
        list(utilities.keys())
    )
    func_name = sys.argv[1]
    utilities[func_name]()
