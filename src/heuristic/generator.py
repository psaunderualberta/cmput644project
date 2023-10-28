import numpy as np
from src.heuristic.expressions import Binary, Unary, Terminal, Number
from src.utility.constants import MAX_TREE_SIZE, NORMALIZED_COLUMN_NAMES


def random_heuristic(tree_size=np.random.randint(1, MAX_TREE_SIZE)):
    """
    Generate a random heuristic.
    """

    if tree_size < 1:
        tree_size = np.random.randint(1, MAX_TREE_SIZE)

    # Base cases
    if tree_size == 1:
        return np.random.choice([__random_terminal(), __random_number()])
    elif tree_size == 2:
        # With a heuristic size of 2, we can only have unary -> terminal
        # we can't have a binary, since that implies at least 3 terms
        return __random_unary(2)

    if np.random.randint(0, 2) == 0:
        return __random_unary(tree_size)
    return __random_binary(tree_size)


def __random_binary(tree_size):

    # Generate a random binary operator
    op = np.random.choice(["plus", "sub", "mul", "div", "max", "min"])
    # Generate a random left subtree
    left = random_heuristic(np.random.randint(1, tree_size - 1))
    # Generate a random right subtree
    right = random_heuristic(tree_size - left.size() - 1)
    # Create the binary heuristic
    return Binary(op, left, right)


def __random_unary(tree_size):
    # Generate a random unary operator
    op = np.random.choice(["neg", "abs", "sqrt", "sqr"])
    # Generate a random subtree
    right = random_heuristic(tree_size - 1)
    # Create the unary heuristic
    return Unary(op, right)


def __random_terminal():
    # Generate a random terminal
    return Terminal(np.random.choice(NORMALIZED_COLUMN_NAMES))


def __random_number():
    # Generate a random number, rounded to 1 decimal place
    num = np.round(np.random.uniform(-10, 10), 1)
    return Number(num)
