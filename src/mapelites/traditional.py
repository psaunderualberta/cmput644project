from src.mapelites.population import PopulationStorage
import numpy as np
from src.heuristic.parsing import parse_heuristic
from src.heuristic.mutator import mutate_heuristic

class TraditionalPopulation(PopulationStorage):
    def __init__(self, num_elites, num_best_heuristics):
        self.num_best_heuristics = num_best_heuristics
        self.num_elites = num_elites
        self.population = np.array([])
        self.best_heuristics = np.array([])
        self.best_fitnesses = np.array([])
        self.elites = np.array([])

    def insert_heuristic_if_better(self, h, f):
        # the array of elites is no longer valid,
        # and the population is no longer valid
        self.elites = np.array([])
        self.population = np.array([])

        # Append to the record
        self.best_heuristics = np.append(self.best_heuristics, h)
        self.best_fitnesses = np.append(self.best_fitnesses, f)

        # Sort the two arrays according to the fitness value
        sorted_idxs = np.argsort(self.best_fitnesses)
        self.best_heuristics = self.best_heuristics[sorted_idxs]
        self.best_fitnesses = self.best_fitnesses[sorted_idxs]

        # Trim to the size of best heuristics
        self.best_heuristics = self.best_heuristics[self.num_best_heuristics]
        self.best_fitnesses = self.best_fitnesses[self.num_best_fitnesses]
    
    def get_random_heuristic(self, fitnesses):
        if len(self.elites) == 0:
            self.elites = self.__select_elites(fitnesses)

        heuristic = np.random.choice(self.elites)
        parsed = parse_heuristic(heuristic)
        mutated = mutate_heuristic(parsed)
        self.population = np.append(self.population, str(mutated))
        
        return mutated
    
    def get_fitnesses(self):
        return np.array(self.best_fitnesses)
    
    def get_stored_data(self, strip_nan=False, unique=True):
        return self.best_heuristics, self.best_fitnesses
    
    def __select_elites(self, fitnesses):
        sorted_idxs = np.argsort(fitnesses)
        self.elites = self.population[sorted_idxs[:self.num_elites]]
