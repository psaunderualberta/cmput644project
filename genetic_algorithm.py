import pickle
import random
import time

import numpy as np
import pandas as pd
import wandb
from dask import compute, delayed
from dask.distributed import Client, LocalCluster

from src.mapelites.table_array import TableArray
from src.utility.constants import *
from src.mapelites.traditional import TraditionalPopulation
from src.mapelites.population import PopulationStorage

def main():
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)
    print(client.dashboard_link)

    # Get today's date, to the hour, in a format that can be used as a filename
    prefix = "testing"

    config = {
        "SEED": 1337,
        "POPULATION_SRC": SHORTENED_DATA_FILES,
        "POPULATION_SIZE": 20,
        "TIMEOUT": 3 * 60,  # 3 minutes
        "WANDB": False,
        "WANDB_PROJECT": "cmput644project",
        "WANDB_ENTITY": "psaunder",
        "POPULATION_TYPE": "traditional",
        "LOG_FILE": os.path.join("logs", "results", f"{prefix}.txt"),
    }    

    # Create the log file
    os.makedirs(os.path.dirname(config["LOG_FILE"]), exist_ok=True)
    with open(config["LOG_FILE"], "w") as f:
        f.write("Population Number | Heuristic | Fitness | NaN Count | Number of valid samples\n")

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
    resolutions = [11, 6, 4, 3]
    if config["POPULATION_TYPE"] == "mapelites":
        methods = ["size", "depth"]
        ranges = [[0, MAX_TREE_SIZE], [0, MAX_TREE_DEPTH]]
        heuristic_storage: PopulationStorage = TableArray(methods, ranges, resolutions)
    elif config["POPULATION_TYPE"] == "traditional":
        elite_percentage = 0.25
        num_elites = np.ceil(config["POPULATION_SIZE"] * elite_percentage).astype(np.int64)

        # Approximately half of the mapelites table gets filled in
        num_best_heuristics = (MAX_TREE_SIZE ** 2) // 2
        heuristic_storage: PopulationStorage = TraditionalPopulation(num_elites, num_best_heuristics)
    else:
        raise ValueError(f"Heuristic Storage type '{config['POPULATION_TYPE']}' is not recognized.")

    # Create initial population
    population = heuristic_storage.get_next_population([None] * config["POPULATION_SIZE"])

    # Load data
    dfs = [delayed_load_data(file) for file in config["POPULATION_SRC"]]
    delayed_targets = [df[CLASSES_2_Y_COLUMN] for df in dfs]
    delayed_targets = [delayed(lambda x: x.astype(np.float64))(target) for target in delayed_targets]

    evolution_start = time.time()
    population_number = 1
    # While the timeout has not been reached
    while time.time() - evolution_start < config["TIMEOUT"]:
        print("Time left: {:0.3f}s".format(config["TIMEOUT"] - (time.time() - evolution_start)))

        # Execute the population
        # To parallize this, we use dask.delayed and a custom computation of the fitness
        # Without this, we would need to load the entire dataset into memory, which might not be
        # possible with some machines.
        delayed_fitnesses = []
        delayed_nan_counts = []
        delayed_num_samples = []
        for heuristic in population:
            feature_sums = []
            feature_sum_squares = []
            target_sum = []
            target_sum_squares = []
            product_sums = []

            # Execute the heuristic on each subset of the dataset
            new_feats = [delayed_execute_heuristic(df, heuristic) for df in dfs]

            # Calculate the mean, variance, and product sum for each subset
            non_nan_lens = []
            nan_counts = []
            trimmed_feats = []
            trimmed_targets = []
            assert len(new_feats) == len(delayed_targets)
            for new_feat, delayed_target in zip(new_feats, delayed_targets):
                trimmed_feat, trimmed_target, new_feat_len, nan_count = delayed_trim_nans(new_feat, delayed_target)
                trimmed_feats.append(trimmed_feat)
                trimmed_targets.append(trimmed_target)
                non_nan_lens.append(new_feat_len)
                nan_counts.append(nan_count)
            
            feature_sums = [delayed_sum(new_feat) for new_feat in trimmed_feats]
            feature_sum_squares = [delayed_sum_squares(new_feat) for new_feat in trimmed_feats]
            target_sum = [delayed_sum(delayed_target) for delayed_target in trimmed_targets]
            target_sum_squares = [delayed_sum_squares(delayed_target) for delayed_target in trimmed_targets]
            product_sums = [delayed_product_sum(new_feat, delayed_target) for new_feat, delayed_target in zip(trimmed_feats, trimmed_targets)]

            # Sum each vector
            feature_sum = delayed(np.sum)(feature_sums)
            feature_sum_square = delayed(np.sum)(feature_sum_squares)
            target_sum = delayed(np.sum)(target_sum)
            target_sum_square = delayed(np.sum)(target_sum_squares)
            product_sum = delayed(np.sum)(product_sums)

            # Get the # of non-nan samples
            num_samples = delayed(np.sum)(non_nan_lens)
            delayed_num_samples.append(num_samples)
            delayed_nan_counts.append(delayed(np.sum)(nan_counts))
            
            # Calculate the correlation between the heuristic and the target
            # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
            numerator = num_samples * product_sum - (feature_sum * target_sum)
            lhs_denominator = delayed(np.sqrt)(num_samples * feature_sum_square - feature_sum ** 2)
            rhs_denominator = delayed(np.sqrt)(num_samples * target_sum_square - target_sum ** 2)

            # Calculate the fitness of the heuristic
            corr = numerator / (lhs_denominator * rhs_denominator)
            delayed_fitnesses.append(delayed(abs)(corr))

        start = time.time()
        assert len(delayed_fitnesses) == len(delayed_nan_counts) == len(delayed_num_samples)
        fitnesses, nan_counts, num_samples = compute(delayed_fitnesses, delayed_nan_counts, delayed_num_samples)
        fitnesses, nan_counts, num_samples = np.array(fitnesses), np.array(nan_counts), np.array(num_samples)
        assert len(fitnesses) == len(nan_counts) == len(delayed_num_samples)
        fitnesses = np.where(np.isnan(fitnesses) | np.isinf(fitnesses), 0, fitnesses)
        fitnesses = np.where(nan_counts > MAX_NAN_PERCENTAGE * num_samples, 0, fitnesses)
        print("Time to compute fitnesses: {}".format(time.time() - start))

        # Insert the population into the storage
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

        # Write current population and results to log file
        with open(config["LOG_FILE"], "a") as file:
            for heuristic, fitness, nan_count, num_sample in zip(population, fitnesses, nan_counts, num_samples):
                file.write(f"{population_number} | {heuristic} | {fitness} | {nan_count} | {num_sample}\n")

        # Select new population
        population = heuristic_storage.get_next_population(fitnesses)
        population_number += 1

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
    dir_name = os.path.join(".", "artifacts", "local")
    os.makedirs(dir_name, exist_ok=True)
    fname = os.path.join(os.path.join(dir_name, "tables.pkl"))
    with open(fname, "wb") as f:
        pickle.dump(heuristic_storage, f)

    if config["WANDB"]:
        artifact = wandb.Artifact("map-elites", type="dataset")
        artifact.add_file(fname)
        wandb.log_artifact(artifact)

        # Remove the file once it has been uploaded to wandb
        os.remove(fname)


@delayed(nout=4)
def delayed_trim_nans(feats, targets):
    assert len(feats) == len(targets)
    valid_idxs = ~(np.isnan(feats) | np.isinf(feats))
    feats = feats[valid_idxs]
    targets = targets[valid_idxs]
    assert len(feats) == len(targets)
    return feats, targets, len(feats), np.sum(~valid_idxs)


@delayed
def delayed_sum(feats):
    return np.sum(feats)


@delayed
def delayed_sum_squares(feats):
    return np.sum(feats ** 2)


@delayed
def delayed_product_sum(feats, targets):
    return np.dot(feats, targets)


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
