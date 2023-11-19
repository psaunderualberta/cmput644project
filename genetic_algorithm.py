from src.utility.util import load_data
from src.utility.constants import *
from src.heuristic.generator import random_heuristic
from src.heuristic.mutator import mutate_heuristic
from src.mapelites.table_array import TableArray
import numpy as np
import wandb
import random
import time
import pickle

def main():
    config = {
        "SEED": 42,
        "POPULATION_SIZE": 20,
        "TIMEOUT": 12 * 60 * 60,  # 12 hours
        "WANDB": True,
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
    data = load_data(COMBINED_DATA_FILES)
    targets = data[CLASSES_2_Y_COLUMN].to_numpy()
    
    # Create initial population
    population = [random_heuristic(MAX_TREE_SIZE) for _ in range(config["POPULATION_SIZE"])]
    
    start = time.time()
    # While the timeout has not been reached
    while time.time() - start < config["TIMEOUT"]:
        print("Time left: {:0.3f}s".format(config["TIMEOUT"] - (time.time() - start)))

        # Evaluate the population
        fitnesses = [compute_fitness(h, data, targets) for h in population]

        # Insert the population into MAP-Elites
        for heuristic, fitness in zip(population, fitnesses):
            tables.insert_heuristic_if_better(heuristic, fitness)

        # log Statistics to wandb
        if config["WANDB"]:
            # Population specific statistics
            mapelites_fitnesses = tables.get_fitnesses(len(resolutions) - 1)
            wandb.log({
                "Mean Population Fitness": np.nanmean(fitnesses),
                "Max Population Fitness": np.nanmax(fitnesses),
                "Min Population Fitness": np.nanmin(fitnesses),
                "Std Population Fitness": np.nanstd(fitnesses),

                # MAP-Elites specific statistics
                "Mean MAP-Elites Fitness": np.nanmean(mapelites_fitnesses),
                "Max MAP-Elites Fitness": np.nanmax(mapelites_fitnesses),
                "Min MAP-Elites Fitness": np.nanmin(mapelites_fitnesses),
                "Std MAP-Elites Fitness": np.nanstd(mapelites_fitnesses),
            })

        # Select new population
        population = [tables.get_random_heuristic() for _ in range(config["POPULATION_SIZE"])]

        # Mutate new population
        for i, heuristic in enumerate(population):
            population[i] = mutate_heuristic(heuristic)

    # Log final statistics to wandb
    heuristics, fitnesses = tables.get_stored_data(True)
    best_idx = np.argmax(fitnesses)
    best_heuristic = heuristics[best_idx]
    best_fitness = fitnesses[best_idx]
    print("Best Fitness: {:0.4f}".format(best_fitness))
    print("Best Heuristic: {}".format(best_heuristic))
    if config["WANDB"]:
        wandb.log({
            "Best Fitness": str(best_fitness),
            "Best Heuristic": str(best_heuristic),
            "All Heuristics": list(map(str, heuristics)),
            "All Fitnesses": fitnesses,
        })

    # Save the 'TablesArray' object to a pickle file, then upload to wandb
    if config["WANDB"]:
        fname = os.path.join("tables.pkl")
        with open(fname, "wb") as f:
            pickle.dump(tables, f)

        artifact = wandb.Artifact("map-elites", type="dataset")
        artifact.add_file("tables.pkl")
        wandb.log_artifact(artifact)
        os.remove(fname)

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
