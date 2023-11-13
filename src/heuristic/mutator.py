from src.heuristic.expressions import Binary, Unary, Terminal, Number
from src.heuristic.generator import random_heuristic
from src.utility.constants import MAX_TREE_SIZE
import numpy as np

from copy import deepcopy


def mutate_heuristic(heuristic):
    """
    Mutate a heuristic by randomly replacing a subtree with a new subtree.
    """
    heuristic = deepcopy(heuristic)
    mut_probability = 1 / heuristic.size()

    mutated = False
    while not mutated:
        mutated, heuristic = __mutate_heuristic(heuristic, mut_probability)

    return heuristic


def __mutate_heuristic(heuristic, mut_probability, max_tree_size=MAX_TREE_SIZE):
    """
    Mutate a heuristic by randomly replacing a subtree with a new subtree.
    """
    new_tree_size = np.random.randint(1, max_tree_size + 1)

    if np.random.random() < mut_probability:
        return True, random_heuristic(new_tree_size)

    if isinstance(heuristic, Binary):
        right_size = heuristic.right.size()
        mutated, heuristic.left = __mutate_heuristic(
            heuristic.left, mut_probability, max_tree_size - right_size - 1
        )
        if mutated:
            return True, heuristic

        left_size = heuristic.left.size()
        mutated, heuristic.right = __mutate_heuristic(
            heuristic.right, mut_probability, max_tree_size - left_size - 1
        )
        return mutated, heuristic

    if isinstance(heuristic, Unary):
        mutated, heuristic.right = __mutate_heuristic(
            heuristic.right, mut_probability, max_tree_size - 1
        )
        return mutated, heuristic

    return False, heuristic
