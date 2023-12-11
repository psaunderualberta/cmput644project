import os
import sys

import numpy as np
import pandas as pd
from sympy import *
init_printing()
import pickle

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
        fitness = np.round(df.loc[i]["Fitness"], 3)
        simplified = latex(simplifier.symbolic(h, False), full_prec=False, mul_symbol='dot')
        size = parsed_h.size()
        depth = parsed_h.depth()
        print(f"\t{i+1} & ${simplified}$ & {fitness} & {size} & {depth} \\\\")


def mapelites_2_latex():
    assert (
        len(sys.argv) == 3
    ), "Usage of top_k_to_latex_table: `python utilities.py mapelites_2_latex <pkl_file> "

    fname = sys.argv[2]
    assert os.path.isfile(fname), f"Error: file '{fname}' could not be found."


    # Load the map-elites table
    with open(fname, "rb") as f:
        tables = pickle.load(f)

    # Get unique heuristics
    heuristics, fitnesses = tables.get_stored_data(strip_nan=True, unique=True)

    # Sort based on fitness value
    indxs = np.flip(np.argsort(fitnesses))[:20]
    heuristics = heuristics[indxs]
    fitnesses = fitnesses[indxs]

    # Convert to simplified format
    simplifier = Simplifier()
    for i, (h, f) in enumerate(zip(heuristics, fitnesses)):
        parsed_h = parse_heuristic(h)
        fitness = np.round(f, 3)
        size = parsed_h.size()
        depth = parsed_h.depth()
        # simplified = latex(simplifier.symbolic(h, False), full_prec=False, mul_symbol='dot')
        print(f"\t{i+1} & {fitness} & {size} & {depth} \\\\")


if __name__ == "__main__":
    utilities = {
        top_k_to_latex_table.__name__: top_k_to_latex_table,
        mapelites_2_latex.__name__: mapelites_2_latex,
    }
    assert (
        len(sys.argv) > 1
    ), "Error: Must specify a utility to use. Valid utilities are: " + str(
        list(utilities.keys())
    )
    func_name = sys.argv[1]
    utilities[func_name]()
