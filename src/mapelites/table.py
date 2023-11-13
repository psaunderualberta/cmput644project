import numpy as np
from src.heuristic.generator import random_heuristic


class Table:
    def __init__(self, method_names, ranges, resolution):
        """Create a table for MAP-Elites"""
        # Check that the ranges are valid
        ranges = np.array(ranges)
        assert ranges.shape == (len(method_names), 2)
        assert np.all(ranges[:, 0] < ranges[:, 1])
        assert np.all(ranges[:, 0] >= 0)
        assert np.min(ranges[:, 1] - ranges[:, 0]) + 1 >= resolution

        # Create bins for each method
        self.bins = np.linspace(ranges[:, 0], ranges[:, 1], resolution).T

        # Create the tables
        table_shape = [resolution] * len(method_names)
        self.heuristics = np.full(table_shape, "", dtype=object)
        self.fitnesses = np.full(table_shape, np.NaN, dtype=np.float64)
        self.method_names = method_names
        self.resolution = resolution

    def get_random_heuristic(self):
        """Get a random heuristic from the table"""
        try:
            lengths = np.vectorize(lambda t: len(str(t)))(self.heuristics)
            return np.random.choice(
                self.heuristics[lengths > 0].flatten()
            )
        except ValueError:
            return random_heuristic()

    def insert_heuristic_if_better(self, heuristic, fitness):
        """Insert a heuristic into the table if it is better than the current heuristic
        at the same indices"""
        indices = self.get_indices(heuristic)
        old_heuristic = self.heuristics.item(*indices)
        old_fitness = self.fitnesses.item(*indices)
        if np.isnan(old_fitness) or fitness > old_fitness:
            self.heuristics.itemset(*indices, heuristic)
            self.fitnesses.itemset(*indices, fitness)

        return indices

    def get_indices(self, heuristic):
        """Get the indices of the table for a heuristic"""
        indices = []
        for i, method_name in enumerate(self.method_names):
            result = self.__execute_method(heuristic, method_name)
            index = max(0, np.digitize(result, self.bins[i]) - 1)
            indices.append(index)

        return np.array(indices)

    def __execute_method(self, heuristic, method_name):
        """Execute a method on a heuristic, e.g. `heuristic.size()`"""
        try:
            method = getattr(heuristic, method_name)
        except AttributeError:
            raise ValueError(
                f"Heuristic {heuristic} does not have method {method_name}"
            )

        return method()
