import numpy as np
from src.mapelites.table import Table


class TableArray:
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
        indices = self.get_indices(heuristic)
        for table in self.tables:
            table.insert_heuristic_if_better(heuristic, fitness)

        return indices

    def get_random_heuristic(self):
        """Get a random heuristic from a random table"""
        table = np.random.choice(self.tables)
        return table.get_random_heuristic()
