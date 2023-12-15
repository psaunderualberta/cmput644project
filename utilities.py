import os
import sys

import numpy as np
import pandas as pd
from sympy import *
init_printing()
import pickle

from src.heuristic.parsing import parse_heuristic
from src.heuristic.simplifier import Simplifier
from src.utility.constants import MAPELITES_RESULTS, TRADITIONAL_RESULTS

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
        print(f"\t{h} & {i+1} & ${simplified}$ & {fitness} & {size} & {depth} \\\\")


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
    indxs = np.flip(np.argsort(fitnesses))
    heuristics = heuristics[indxs]
    fitnesses = fitnesses[indxs]

    # Convert to simplified format
    simplifier = Simplifier()
    for i, (h, f) in enumerate(zip(heuristics[13:21], fitnesses[13:21])):
        parsed_h = parse_heuristic(h)
        fitness = np.round(f, 3)
        size = parsed_h.size()
        depth = parsed_h.depth()
        simplified = latex(simplifier.symbolic(h, False), full_prec=False, mul_symbol='dot')
        print(h)
        print(f"\t{i+1} & ${simplified}$ & {fitness} & {size} & {depth} \\\\")


def mapelites_vs_traditional():
    """Determine whether a run is a traditional run or a map-elites run.
    """
    artifact_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "artifacts")
    mapelites_files = glob(
        os.path.join(
            artifact_dir,
            "**",
            "tables.pkl",
        )
    )

    for fname in mapelites_files:
        # Load the tables
        with open(fname, "rb") as f:
            tables = pickle.load(f)
        heuristics, _ = tables.get_stored_data(strip_nan=True, unique=True)

        # get the depth and size of each heuristic
        depths = [parse_heuristic(h).depth() for h in heuristics]
        sizes = [parse_heuristic(h).size() for h in heuristics]

        # If there is a non-unique pair of depth and size, then this is a traditional run
        # otherwise, it is a map-elites run
        if len(set(zip(depths, sizes))) != len(depths):
            print("Traditional run: {}".format(fname))
        else:
            print("Map-Elites run: {}".format(fname))


def get_scores_in_table():
    if len(sys.argv) != 3:
        print("Usage: python utilities.py get_scores_in_table {mapelites | traditional}")
        exit()
    
    if sys.argv[2] == "mapelites":
        folders = MAPELITES_RESULTS
    elif sys.argv[2] == "traditional":
        folders = TRADITIONAL_RESULTS
    else:
        print("Usage: python utilities.py get_scores_in_table {mapelites | traditional}")
        exit()
    
    cols = ["Accuracy", "Precision", "Recall", "F1"]
    total_df = pd.DataFrame(columns=cols, index=range(len(folders)), dtype=np.float64)

    print("Run ID & Accuracy & Precision & Recall & F1")
    for i, folder in enumerate(sorted(folders)):
        score_file = os.path.join(folder, "scores.csv")
        assert os.path.isfile(score_file), "Error: file '{}' could not be found.".format(score_file)
        df = pd.read_csv(score_file).T
        df.columns = df.iloc[0]
        df = df.iloc[1:].astype(np.float64)
        df = df.round(3) * 100
        print("{} & {} & {} & {} & {} \\\\".format(i + 1, *df.iloc[0]))
        total_df.iloc[i] = df.iloc[0]
    
    # Print each column's average and std dev as 'mean \pm std dev'
    print("\midrule")
    last_row = ["Mean $\pm$ Std Dev"]
    for col in total_df.columns:
        mean = total_df[col].mean().round(2)
        std = total_df[col].std().round(2)
        last_row.append("{} $\pm$ {}".format(mean, std))
    
    print(' & '.join(last_row) + " \\\\")


if __name__ == "__main__":
    utilities = {
        top_k_to_latex_table.__name__: top_k_to_latex_table,
        mapelites_2_latex.__name__: mapelites_2_latex,
        mapelites_vs_traditional.__name__: mapelites_vs_traditional,
        get_scores_in_table.__name__: get_scores_in_table,
    }
    assert (
        len(sys.argv) > 1
    ), "Error: Must specify a utility to use. Valid utilities are: " + str(
        list(utilities.keys())
    )
    func_name = sys.argv[1]
    utilities[func_name]()
