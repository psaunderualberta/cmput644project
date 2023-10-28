from src.heuristic.expressions import Binary, Unary, Terminal, Number
from src.heuristic.generator import random_heuristic
from src.utility.constants import MAX_TREE_SIZE
import numpy as np

def mutate_heuristic(heuristic):
    """
    Mutate a heuristic by randomly replacing a subtree with a new subtree.
    """
    mutated = False
    mut_probability = 1 / heuristic.size()
    while not mutated:
        mutated, heuristic = __mutate_heuristic(heuristic, mut_probability)

    return heuristic

def __mutate_heuristic(heuristic, mut_probability, max_tree_size=MAX_TREE_SIZE):
    """
    Mutate a heuristic by randomly replacing a subtree with a new subtree.
    """
    if np.random.random() < mut_probability:
        new_tree_size = np.random.randint(1, max_tree_size)
        return True, random_heuristic(new_tree_size)
    elif isinstance(heuristic, Binary):
        mutated, heuristic.left = __mutate_heuristic(heuristic.left, mut_probability)
        if mutated:
            return True, heuristic
        else:
            mutated, heuristic.right = __mutate_heuristic(heuristic.right, mut_probability)
            return mutated, heuristic
    elif isinstance(heuristic, Unary):
        mutated, heuristic.right = __mutate_heuristic(heuristic.right, mut_probability)
        return mutated, heuristic
    else:
        return False, heuristic