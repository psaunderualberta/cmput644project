from src.heuristic.generator import random_heuristic
from src.utility.constants import MAX_TREE_SIZE

def test_random_heuristic_creation():
    num_iters = 100
    for size in range(1, MAX_TREE_SIZE):
        heuristics = set([random_heuristic(size) for _ in range(num_iters)])

        # Check that the size of the heuristics is correct
        assert all(h.size() == size for h in heuristics)

        # Check and see that there's some randomness in the heuristic generation
        assert len(heuristics) >= 2