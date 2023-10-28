from src.heuristic.expressions import Binary, Unary, Terminal, Number
import pandas as pd

__data = pd.DataFrame(
    {
        "rate": [1.0, 2.0, 3.0],
        "syn_flag_number": [1.0, 2.0, 3.0],
        "ack_count": [1.0, 2.0, 3.0],
    }
)


def test_num_unique_terminals_1():
    assert Number(5.3).num_unique_terminals(__data) == 0


def test_num_unique_terminals_2():
    assert Number(-1.0).num_unique_terminals(__data) == 0


def test_num_unique_terminals_3():
    assert Terminal("rate").num_unique_terminals(__data) == 1


def test_num_unique_terminals_4():
    assert Terminal("syn_flag_number").num_unique_terminals(__data) == 1


def test_num_unique_terminals_5():
    assert Unary("abs", Terminal("ack_count")).num_unique_terminals(__data) == 1


def test_num_unique_terminals_6():
    assert Unary("neg", Terminal("rate")).num_unique_terminals(__data) == 1


def test_num_unique_terminals_7():
    assert (
        Binary("neg", Terminal("ack_count"), Terminal("rate")).num_unique_terminals(
            __data
        )
        == 2
    )


def test_num_unique_terminals_8():
    assert (
        Binary(
            "neg", Terminal("ack_count"), Terminal("ack_count")
        ).num_unique_terminals(__data)
        == 1
    )


def test_num_unique_terminals_9():
    assert (
        Binary(
            "neg", Terminal("ack_count"), Unary("abs", Terminal("ack_count"))
        ).num_unique_terminals(__data)
        == 1
    )


def test_num_unique_terminals_10():
    assert (
        Binary(
            "neg",
            Terminal("ack_count"),
            Binary("plus", Terminal("ack_count"), Terminal("rate")),
        ).num_unique_terminals(__data)
        == 2
    )


def test_num_unique_terminals_11():
    assert (
        Binary(
            "neg",
            Number(1.0),
            Binary("plus", Terminal("ack_count"), Terminal("ack_count")),
        ).num_unique_terminals(__data)
        == 1
    )
