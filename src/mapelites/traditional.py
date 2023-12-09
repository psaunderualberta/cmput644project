from src.mapelites.population import PopulationStorage
import numpy as np
from src.heuristic.parsing import parse_heuristic
from src.heuristic.mutator import mutate_heuristic
from src.heuristic.generator import random_heuristic

class TraditionalPopulation(PopulationStorage):
    def __init__(self, num_elites, num_best_heuristics):
        self.num_best_heuristics = num_best_heuristics
        self.num_elites = num_elites
        self.population = np.array([])
        self.best_heuristics = np.array([])
        self.best_fitnesses = np.array([])
        self.elites = np.array([])

    def insert_heuristic_if_better(self, h, f):
        # Convert heuristic to string (if it isn't already)
        h = str(h)

        # Append to the record
        self.best_heuristics = np.append(self.best_heuristics, h)
        self.best_fitnesses = np.append(self.best_fitnesses, f)

        # Sort the two arrays according to the fitness value (descending order)
        sorted_idxs = np.argsort(-self.best_fitnesses)
        self.best_heuristics = self.best_heuristics[sorted_idxs]
        self.best_fitnesses = self.best_fitnesses[sorted_idxs]

        # Trim to the size of best heuristics
        self.best_heuristics = self.best_heuristics[:self.num_best_heuristics]
        self.best_fitnesses = self.best_fitnesses[:self.num_best_heuristics]
    
    def get_next_population(self, fitnesses):
        if all(f is None for f in fitnesses):
            self.population = np.array([str(random_heuristic()) for _ in fitnesses])
            return list(map(parse_heuristic, self.population))

        self.__select_elites(fitnesses)

        self.population = self.elites.copy()
        while len(self.population) < len(fitnesses):
            heuristic = np.random.choice(self.elites)
            parsed = parse_heuristic(heuristic)
            mutated = mutate_heuristic(parsed)
            self.population = np.append(self.population, str(mutated))
        
        return list(map(parse_heuristic, self.population))
    
    def get_fitnesses(self, _):
        return np.array(self.best_fitnesses)
    
    def get_stored_data(self, strip_nan=False, unique=True):
        return self.best_heuristics, self.best_fitnesses
    
    def __select_elites(self, fitnesses):
        sorted_idxs = np.argsort(fitnesses)
        self.elites = self.population[sorted_idxs[:self.num_elites]]
