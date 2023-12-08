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

def main():
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)
    print(client.dashboard_link)

    config = {
        "SEED": 69,
        "POPULATION_SRC": SHORTENED_DATA_FILES,
        "POPULATION_SIZE": 20,
        "TIMEOUT": 20,  # 12 hours
        "WANDB": False,
        "WANDB_PROJECT": "cmput644project",
        "WANDB_ENTITY": "psaunder",
        "POPULATION_TYPE": "traditional"
    }

    # TODO: Pretty print config

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

    # Create heuristic_storage
    if config["POPULATION_TYPE"] == "mapelites":
        methods = ["size", "depth"]
        ranges = [[0, MAX_TREE_SIZE], [0, MAX_TREE_DEPTH]]
        resolutions = [11, 6, 4, 3]
        heuristic_storage: PopulationStorage = TableArray(methods, ranges, resolutions)
    elif config["POPULATION_TYPE"] == "traditional":
        elite_percentage = 0.25
        num_elites = np.ceil(config["POPULATION_SIZE"] * elite_percentage).astype(np.int64)

        # Approximately half of the mapelites table gets filled in
        num_best_heuristics = (MAX_TREE_SIZE ** 2) // 2
        heuristic_storage: PopulationStorage = TraditionalPopulation(num_elites, num_best_heuristics)
    else:
        raise ValueError(f"Heuristic Storage type '{config['POPULATION_TYPE']}' is not recognized.")

    # Load data
    dfs = [delayed_load_data(file) for file in config["POPULATION_SRC"]]
    targets = [df[CLASSES_2_Y_COLUMN] for df in dfs]

    # Create initial population
    population = heuristic_storage.get_next_population([None] * config["POPULATION_SIZE"])

    evolution_start = time.time()
    # While the timeout has not been reached
    while time.time() - evolution_start < config["TIMEOUT"]:
        print("Time left: {:0.3f}s".format(config["TIMEOUT"] - (time.time() - evolution_start)))

        # Execute the population
        delayed_features = []
        for heuristic in population:
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

        # Insert the population into MAP-Elites
        for heuristic, fitness in zip(population, fitnesses):
            heuristic_storage.insert_heuristic_if_better(heuristic, fitness)

        # log Statistics to wandb
        if config["WANDB"]:
            # Population specific statistics
            mapelites_fitnesses = heuristic_storage.get_fitnesses(len(resolutions) - 1)
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
        population = heuristic_storage.get_next_population(fitnesses)
        print(population)

    # Log final statistics to wandb
    heuristics, fitnesses = heuristic_storage.get_stored_data(True)
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

    # Save the 'heuristic_storageArray' object to a pickle file, then upload to wandb
    if config["WANDB"]:
        os.mkdir("artifacts", exist_ok=True)
        fname = os.path.join(os.path.join("heuristic_storage.pkl"))
        with open(fname, "wb") as f:
            pickle.dump(heuristic_storage, f)

        artifact = wandb.Artifact("map-elites", type="dataset")
        artifact.add_file(fname)
        wandb.log_artifact(artifact)
        os.remove(fname)


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
