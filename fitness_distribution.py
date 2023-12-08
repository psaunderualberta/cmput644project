import pickle
import random
import time

import numpy as np
import pandas as pd
import wandb
from dask import compute, delayed
from dask.distributed import Client, LocalCluster, wait

from src.heuristic.generator import random_heuristic
from src.heuristic.mutator import mutate_heuristic
from src.mapelites.table_array import TableArray
from src.utility.constants import *
from src.utility.util import load_data
from src.mapelites.traditional import TraditionalPopulation
from src.mapelites.population import PopulationStorage
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)
    print(client.dashboard_link)

    config = {
        "NUM_HEURISTICS": 100,
        "POPULATION_SRC": SHORTENED_DATA_FILES,
    }

    # Load the data
    dfs = [delayed_load_data(file) for file in config["POPULATION_SRC"]]
    targets = [df[CLASSES_2_Y_COLUMN] for df in dfs]

    heuristics = [random_heuristic() for _ in range(config["NUM_HEURISTICS"])]

    # Execute the population
    delayed_features = []
    for heuristic in heuristics:
        new_feats = [delayed_execute_heuristic(df, heuristic) for df in dfs]
        delayed_features.append(new_feats)

    # Evaluate the population
    fitnesses = [
        delayed_compute_fitness(delayed_feature, targets)
        for delayed_feature in delayed_features
    ]

    start = time.time()
    fitnesses = compute(*fitnesses)
    print("Time to compute fitnesses: {}".format(time.time() - start))

    # Trim NaN values
    fitnesses = np.array(fitnesses)
    old_len = len(fitnesses)
    fitnesses = fitnesses[~np.isnan(fitnesses)]
    new_len = len(fitnesses)
    nan_percentage = np.round((old_len - new_len) / old_len * 100, 2)

    sns.histplot(fitnesses)
    title = f"{config['NUM_HEURISTICS']} heuristics: {nan_percentage}% produced NaN fitnesses"
    plt.title(title)

    file_location = os.path.dirname(os.path.realpath(__file__))
    plot_dir = os.path.join(file_location, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fname = "hist.pdf"
    plt.savefig(os.path.join(plot_dir, fname))


@delayed
def delayed_compute_fitness(features, targets):
    features = np.concatenate(features)
    targets = np.concatenate(targets)
    # Remove nan values
    non_nan_idxs = ~np.isnan(features)
    non_nan_values = features[non_nan_idxs]
    non_nan_targets = targets[non_nan_idxs]

    # Calculate fitness
    return abs(np.corrcoef(non_nan_values, non_nan_targets)[0][1])


@delayed
def delayed_load_data(file):
    df = pd.read_parquet(file)
    df.columns = list(
        map(
            lambda col: NORMALIZED_COLUMN_NAMES_MAPPING[col]
            if col in NORMALIZED_COLUMN_NAMES_MAPPING
            else col,
            df.columns,
        )
    )
    return df


@delayed
def delayed_execute_heuristic(df, heuristic):
    return heuristic.execute(df)


if __name__ == "__main__":
    main()