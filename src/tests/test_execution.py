import numpy as np
import pandas as pd

from src.heuristic.transformer import Binary, Number, Terminal, Unary

__data = pd.DataFrame(
    {
        "rate": [1.0, 2.0, 3.0],
        "syn_flag_number": [1.0, 2.0, 3.0],
        "ack_count": [1.0, 2.0, 3.0],
    }
)

## Numbers
def test_executor_number_1():
    assert np.all(Number(5.3).execute(__data) == np.array([5.3, 5.3, 5.3]))


def test_executor_number_2():
    assert np.all(Number(-1.0).execute(__data) == np.array([-1.0, -1.0, -1.0]))


## Terminals
def test_executor_terminal_1():
    assert np.all(Terminal("rate").execute(__data) == np.array([1.0, 2.0, 3.0]))


def test_executor_terminal_2():
    assert np.all(
        Terminal("syn_flag_number").execute(__data) == np.array([1.0, 2.0, 3.0])
    )


def test_executor_terminal_3():
    assert np.all(Terminal("ack_count").execute(__data) == np.array([1.0, 2.0, 3.0]))


## Unaries
def test_executor_unary_1():
    h = Unary("neg", Terminal("rate"))
    assert np.all(h.execute(__data) == np.array([-1.0, -2.0, -3.0]))


def test_executor_unary_2():
    h = Unary("abs", Number(-1.0))
    assert np.all(h.execute(__data) == np.array([1.0, 1.0, 1.0]))


# Special case of sqrt
def test_executor_unary_3():
    h = Unary("sqrt", Number(-4.0))
    assert np.all(h.execute(__data) == np.array([-2.0, -2.0, -2.0]))


def test_executor_unary_4():
    h = Unary("neg", Unary("sqrt", Terminal("ack_count")))
    assert np.all(h.execute(__data) == np.array([-1.0, -np.sqrt(2), -np.sqrt(3)]))


## Binaries
def test_executor_binary_1():
    h = Binary("plus", Terminal("rate"), Number(1.0))
    assert np.all(h.execute(__data) == np.array([2.0, 3.0, 4.0]))


def test_executor_binary_2():
    h = Binary("sub", Terminal("rate"), Number(1.0))
    assert np.all(h.execute(__data) == np.array([0.0, 1.0, 2.0]))


def test_executor_binary_3():
    h = Binary("mul", Terminal("rate"), Number(2.0))
    assert np.all(h.execute(__data) == np.array([2.0, 4.0, 6.0]))


## Deep expressions
def test_executor_deep_1():
    h = Binary("plus", Terminal("rate"), Binary("mul", Terminal("rate"), Number(2.0)))
    assert np.all(h.execute(__data) == np.array([3.0, 6.0, 9.0]))


def test_executor_deep_2():
    h = Binary(
        "max", Unary("neg", Terminal("rate")), Unary("abs", Terminal("ack_count"))
    )
    assert np.all(h.execute(__data) == np.array([1.0, 2.0, 3.0]))
