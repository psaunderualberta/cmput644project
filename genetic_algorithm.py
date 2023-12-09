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
        "POPULATION_SRC": COMBINED_DATA_FILES,
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
    delayed_targets = [df[CLASSES_2_Y_COLUMN] for df in dfs]
    delayed_targets = [delayed(lambda x: x.astype(np.float64))(target) for target in delayed_targets]

    # Create initial population
    population = heuristic_storage.get_next_population([None] * config["POPULATION_SIZE"])

    evolution_start = time.time()
    # While the timeout has not been reached
    while time.time() - evolution_start < config["TIMEOUT"]:
        print("Time left: {:0.3f}s".format(config["TIMEOUT"] - (time.time() - evolution_start)))

        # Execute the population
        # To parallize this, we use dask.delayed and a custom computation of the fitness
        delayed_fitnesses = []
        delayed_nan_counts = []
        for heuristic in population:
            delayed_feature_means = []
            delayed_feature_variances = []
            delayed_target_means = []
            delayed_target_variances = []
            delayed_products = []

            # Execute the heuristic on each subset of the dataset
            new_feats = [delayed_execute_heuristic(df, heuristic) for df in dfs]

            # Calculate the mean, variance, and product sum for each subset
            non_nan_lens = []
            nan_counts = []
            assert len(new_feats) == len(delayed_targets)
            for new_feat, delayed_target in zip(new_feats, delayed_targets):
                new_feat, delayed_target, new_feat_len, nan_count = delayed_trim_nans(new_feat, delayed_target)
                nan_counts.append(nan_count)
                non_nan_lens.append(new_feat_len)
                delayed_feature_mean, delayed_target_mean = delayed_means(new_feat, delayed_target)
                delayed_feature_var, delayed_target_var = delayed_variances(new_feat, delayed_target)
                delayed_feature_means.append(delayed_feature_mean)
                delayed_target_means.append(delayed_target_mean)
                delayed_feature_variances.append(delayed_feature_var)
                delayed_target_variances.append(delayed_target_var)
                delayed_products.append(delayed_product_sum(new_feat, delayed_target))

            # Calculate the overall mean and variance of the heuristic across the dataset
            feature_mean, feature_var = delayed_pooled_mean_and_var(delayed_feature_means, delayed_feature_variances, non_nan_lens)
            target_mean, target_var = delayed_pooled_mean_and_var(delayed_target_means, delayed_target_variances, non_nan_lens)
            feature_std = feature_var ** 0.5
            target_std = target_var ** 0.5

            # Calculate the correlation between the heuristic and the target
            # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
            product_sum = delayed(np.sum)(delayed_products)
            num_samples = delayed(np.sum)(non_nan_lens)
            delayed_nan_counts.append(delayed(np.sum)(nan_counts))
            corr = (product_sum - num_samples * feature_mean * target_mean) / ((num_samples - 1) * feature_std * target_std)
            delayed_fitnesses.append(delayed(abs)(corr))

        assert len(delayed_fitnesses) == len(delayed_nan_counts)
        print(len(delayed_fitnesses), len(delayed_nan_counts))
        start = time.time()
        fitnesses, nan_counts = compute(delayed_fitnesses, delayed_nan_counts)
        print(len(fitnesses), len(nan_counts))
        print(fitnesses, nan_counts)
        assert len(fitnesses) == len(nan_counts)
        fitnesses = np.where(np.isnan(fitnesses) | np.isinf(fitnesses), 0, fitnesses)

        print(fitnesses)
        print("Time to compute fitnesses: {}".format(time.time() - start))
        exit()

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

# def compute_fitness(features, targets):
#     # Remove nan values
#     non_nan_idxs = ~np.isnan(features)
#     non_nan_values = features[non_nan_idxs]
#     non_nan_targets = targets[non_nan_idxs]

#     # Calculate fitness
#     return abs(np.corrcoef(non_nan_values, non_nan_targets)[0][1])

@delayed(nout=4)
def delayed_trim_nans(feats, targets):
    valid_idxs = ~(np.isnan(feats) | np.isinf(feats))
    feats = feats[valid_idxs]
    targets = targets[valid_idxs]
    return feats, targets, len(feats), np.sum(~valid_idxs)

@delayed(nout=2)
def delayed_means(feats, targets):
    return (
        np.mean(feats),
        np.mean(targets)
    )


@delayed(nout=2)
def delayed_variances(feats, targets):
    return (
        np.var(feats),
        np.var(targets)
    )


@delayed(nout=2)
def delayed_pooled_mean_and_var(means, variances, lens):
    means = np.array(means)
    variances = np.array(variances)
    lens = np.array(lens)

    return (
        np.sum(means * lens, axis=0) / np.sum(lens),
        np.sum((lens - 1) * variances, axis=0) / (np.sum(lens) - 1),
    )

@delayed
def delayed_product_sum(feats, targets):
    print(feats.shape, targets.shape)
    print(np.dot(feats, targets).shape)
    print(np.dot(feats, targets))
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
