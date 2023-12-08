from functools import reduce

import numpy as np

from src.mapelites.table import Table
from src.mapelites.population import PopulationStorage


class TableArray(PopulationStorage):
    def __init__(self, method_names, ranges, resolutions):
        """Create a table for MAP-Elites"""
        ranges = np.array(ranges)
        assert ranges.shape == (len(method_names), 2)
        assert np.all(ranges[:, 0] < ranges[:, 1])
        assert np.all(ranges[:, 0] >= 0)

        self.tables = [
            Table(method_names, ranges, resolution) for resolution in resolutions
        ]

    def insert_heuristic_if_better(self, heuristic, fitness):
        """Insert a heuristic into the table if it is better than the current heuristic
        at the same indices"""
        for table in self.tables:
            table.insert_heuristic_if_better(heuristic, fitness)

    def get_next_population(self, fitnesses):
        """Creates the next population by sampling from the current mapelites tables"""
        return [
            self.get_random_mutated_heuristic(None) for _ in range(len(fitnesses))
        ]

    def get_random_mutated_heuristic(self, _):
        """Get a random heuristic from a random table"""
        table = np.random.choice(self.tables)
        return table.get_random_heuristic()

    def get_fitnesses(self, idx=None):
        """Get the fitnesses of all heuristics in the table"""
        if idx is None:
            return np.concatenate([table.fitnesses.flatten() for table in self.tables])

        return self.tables[idx].fitnesses.flatten()

    def get_stored_data(self, strip_nan=False, unique=True):
        """Get the heuristics and fitnesses of all heuristics in the table"""
        heuristics = []
        fitnesses = []
        for table in self.tables:
            heuristics.append(table.heuristics.flatten())
            fitnesses.append(table.fitnesses.flatten())
        
        heuristics = np.concatenate(heuristics)
        fitnesses = np.concatenate(fitnesses)

        if strip_nan:
            non_nan_idxs = ~np.isnan(fitnesses)
            heuristics = heuristics[non_nan_idxs]
            fitnesses = fitnesses[non_nan_idxs]
        
        if unique:
            heuristics, unique_idxs = np.unique(heuristics, return_index=True)
            fitnesses = fitnesses[unique_idxs]

        return heuristics, fitnesses
