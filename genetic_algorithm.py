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


def main():
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)
    print(client.dashboard_link)

    config = {
        "SEED": 69,
        "POPULATION_SIZE": 20,
        "TIMEOUT": 12 * 60 * 60,  # 12 hours
        "WANDB": False,
        "WANDB_PROJECT": "cmput644project",
        "WANDB_ENTITY": "psaunder",
    }

    # Set seed
    if "seed" in config:
        np.random.seed(config["seed"])
        random.seed(config["seed"])

    if config["WANDB"]:
        wandb.init(
            config=config,
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
        )

    # Create mapelites tables
    methods = ["size", "depth"]
    ranges = [[0, MAX_TREE_SIZE], [0, MAX_TREE_DEPTH]]
    resolutions = [11, 6, 4, 3]
    tables = TableArray(methods, ranges, resolutions)

    # Load data
    dfs = [delayed_load_data(file) for file in COMBINED_DATA_FILES]
    delayed_targets = [df[CLASSES_2_Y_COLUMN] for df in dfs]
    targets = np.concatenate(compute(*delayed_targets))

    # Create initial population
    population = [
        random_heuristic(MAX_TREE_SIZE) for _ in range(config["POPULATION_SIZE"])
    ]

    start = time.time()
    # While the timeout has not been reached
    while time.time() - start < config["TIMEOUT"]:
        print("Time left: {:0.3f}s".format(config["TIMEOUT"] - (time.time() - start)))

        # Execute the population
        delayed_features = []
        for heuristic in population:
            new_feats = [delayed_execute_heuristic(df, heuristic) for df in dfs]
            concat = delayed(np.concatenate)(new_feats)
            delayed_features.append(concat)

        # Evaluate the population
        fitnesses = [
            delayed_compute_fitness(delayed_feature, targets)
            for delayed_feature in delayed_features
        ]

        start = time.time()
        fitnesses = compute(*fitnesses)
        print("Time to compute fitnesses: {}".format(time.time() - start))
        exit()

        # Insert the population into MAP-Elites
        for heuristic, fitness in zip(population, fitnesses):
            tables.insert_heuristic_if_better(heuristic, fitness)

        # log Statistics to wandb
        if config["WANDB"]:
            # Population specific statistics
            mapelites_fitnesses = tables.get_fitnesses(len(resolutions) - 1)
            wandb.log(
                {
                    "Mean Population Fitness": np.nanmean(fitnesses),
                    "Max Population Fitness": np.nanmax(fitnesses),
                    "Min Population Fitness": np.nanmin(fitnesses),
                    "Std Population Fitness": np.nanstd(fitnesses),
                    # MAP-Elites specific statistics
                    "Mean MAP-Elites Fitness": np.nanmean(mapelites_fitnesses),
                    "Max MAP-Elites Fitness": np.nanmax(mapelites_fitnesses),
                    "Min MAP-Elites Fitness": np.nanmin(mapelites_fitnesses),
                    "Std MAP-Elites Fitness": np.nanstd(mapelites_fitnesses),
                }
            )

        # Select new population
        population = [
            tables.get_random_mutated_heuristic() for _ in range(config["POPULATION_SIZE"])
        ]

    # Log final statistics to wandb
    heuristics, fitnesses = tables.get_stored_data(True)
    best_idx = np.argmax(fitnesses)
    best_heuristic = heuristics[best_idx]
    best_fitness = fitnesses[best_idx]
    print("Best Fitness: {:0.4f}".format(best_fitness))
    print("Best Heuristic: {}".format(best_heuristic))
    if config["WANDB"]:
        wandb.log(
            {
                "Best Fitness": str(best_fitness),
                "Best Heuristic": str(best_heuristic),
                "All Heuristics": list(map(str, heuristics)),
                "All Fitnesses": fitnesses,
            }
        )

    # Save the 'TablesArray' object to a pickle file, then upload to wandb
    if config["WANDB"]:
        os.mkdir("artifacts", exist_ok=True)
        fname = os.path.join(os.path.join("tables.pkl"))
        with open(fname, "wb") as f:
            pickle.dump(tables, f)

        artifact = wandb.Artifact("map-elites", type="dataset")
        artifact.add_file(fname)
        wandb.log_artifact(artifact)
        os.remove(fname)


@delayed
def delayed_compute_fitness(features, targets):
    print(features.shape)
    print(targets.shape)
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
