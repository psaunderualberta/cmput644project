import numpy as np

from src.heuristic.expressions import Binary, Number, Terminal, Unary
from src.mapelites.table import Table


def test_mapelites_table_1():
    method_names = ["size", "depth"]
    ranges = [(0, 100), (0, 100)]
    resolution = 10
    table = Table(method_names, ranges, resolution)
    assert table.method_names == method_names
    assert table.resolution == resolution
    assert table.bins.shape == (len(method_names), resolution)


def test_mapelites_table_2():
    method_names = ["size", "depth"]
    ranges = [(0, 9), (0, 9)]
    resolution = 10
    table = Table(method_names, ranges, resolution)
    heuristic = Binary("plus", Number(1), Number(2))
    indices = table.get_indices(heuristic)
    assert np.all(indices == [3, 2])


def test_mapelites_table_3():
    method_names = ["size", "depth"]
    ranges = [(0, 3), (0, 9)]
    resolution = 3
    table = Table(method_names, ranges, resolution)
    heuristic = Unary("abs", Number(1))
    indices = table.get_indices(heuristic)
    assert np.all(indices == [1, 0])


def test_mapelites_table_4():
    method_names = ["size", "depth"]
    ranges = [(0, 3), (0, 9)]
    resolution = 3
    table = Table(method_names, ranges, resolution)
    heuristic = Unary(
        "abs",
        Unary(
            "abs",
            Unary(
                "abs",
                Unary(
                    "abs", Unary("abs", Unary("abs", Unary("abs", Terminal("rate"))))
                ),
            ),
        ),
    )
    indices = table.get_indices(heuristic)
    assert np.all(indices == [2, 1])


def test_mapelites_table_5():
    method_names = ["size", "depth", "num_unique_terminals"]
    ranges = [(0, 3), (0, 9), (0, 3)]
    resolution = 3
    table = Table(method_names, ranges, resolution)
    heuristic = Binary("plus", Terminal("rate"), Terminal("rate"))
    indices = table.get_indices(heuristic)
    assert np.all(indices == [2, 0, 0])


def test_mapelites_table_6():
    method_names = ["size", "depth", "num_unique_terminals"]
    ranges = [(0, 3), (0, 9), (0, 3)]
    resolution = 3
    table = Table(method_names, ranges, resolution)
    heuristic = Binary("plus", Terminal("rate"), Terminal("Number"))
    indices = table.get_indices(heuristic)
    assert np.all(indices == [2, 0, 1])
