from src.utility.util import load_data
from src.utility.constants import *
from src.heuristic.generator import random_heuristic
from src.heuristic.mutator import mutate_heuristic
from src.mapelites.table_array import TableArray
import numpy as np
import wandb
import random
import time


def main():
    config = {
        "SEED": 42,
        "POPULATION_SIZE": 20,
        "TIMEOUT": 5,
        "WANDB": False,
        "WANDB_PROJECT": "cmput644project",
        "WANDB_ENTITY": "psaunder",
    }

    print(config)

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
    data = load_data(COMBINED_DATA_FILES)
    targets = data[CLASSES_2_Y_COLUMN].to_numpy()
    targets = np.vectorize(lambda x: ATTACK_CLASS if x == ATTACK_CLASS else BENIGN_CLASS)(targets).astype(np.int64)
    targets[0] = 1
    
    # Create initial population
    population = [random_heuristic(MAX_TREE_SIZE) for _ in range(config["POPULATION_SIZE"])]
    
    start = time.time()
    # While the timeout has not been reached
    while time.time() - start < config["TIMEOUT"]:
        print("Time left: {:0.3f}s".format(config["TIMEOUT"] - (time.time() - start)))

        # Evaluate the population
        fitnesses = [compute_fitness(h, data, targets) for h in population]

        # log Statistics to wandb
        if config["WANDB"]:
            # Population specific statistics
            wandb.log({"Mean Population Fitness": np.nanmean(fitnesses)})
            wandb.log({"Max Population Fitness": np.nanmax(fitnesses)})
            wandb.log({"Min Population Fitness": np.nanmin(fitnesses)})
            wandb.log({"Std Population Fitness": np.nanstd(fitnesses)})

            # MAP-Elites specific statistics
            mapelites_fitnesses = tables.get_fitnesses(len(resolutions) - 1)
            wandb.log({"Mean MAP-Elites Fitness": np.nanmean(mapelites_fitnesses)})
            wandb.log({"Max MAP-Elites Fitness": np.nanmax(mapelites_fitnesses)})
            wandb.log({"Min MAP-Elites Fitness": np.nanmin(mapelites_fitnesses)})
            wandb.log({"Std MAP-Elites Fitness": np.nanstd(mapelites_fitnesses)})

        # Insert the population into MAP-Elites
        for heuristic, fitness in zip(population, fitnesses):
            tables.insert_heuristic_if_better(heuristic, fitness)

        # Select new population
        population = [tables.get_random_heuristic() for _ in range(config["POPULATION_SIZE"])]

        # Mutate new population
        for heuristic in population:
            mutate_heuristic(heuristic)
    
    # Log final statistics to wandb
    heuristics, fitnesses = tables.get_stored_data(True)
    best_idx = np.argmax(fitnesses)
    best_heuristic = heuristics[best_idx]
    best_fitness = fitnesses[best_idx]
    print("Best Fitness: {:0.4f}".format(best_fitness))
    print("Best Heuristic: {}".format(best_heuristic))
    if config["WANDB"]:
        wandb.log({"Best Fitness": best_fitness})
        wandb.log({"Best Heuristic": str(best_heuristic)})
        wandb.log({"All Heuristics": heuristics})
        wandb.log({"All Fitnesses": fitnesses})


def compute_fitness(h, data, targets):
    values = h.execute(data)

    # Remove nan values
    non_nan_idxs = ~np.isnan(values)
    non_nan_values = values[non_nan_idxs]
    non_nan_targets = targets[non_nan_idxs]

    # Calculate fitness
    return abs(np.corrcoef(non_nan_values, non_nan_targets)[0][1])


if __name__ == "__main__":
    main()
