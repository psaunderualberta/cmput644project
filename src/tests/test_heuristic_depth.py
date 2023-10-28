from src.heuristic.expressions import Binary, Unary, Terminal, Number


def test_heuristic_depth_1():
    assert Number(5.3).depth() == 1


def test_heuristic_depth_2():
    assert Number(-1.0).depth() == 1


def test_heuristic_depth_3():
    assert Terminal("rate").depth() == 1


def test_heuristic_depth_4():
    assert Terminal("syn_flag_number").depth() == 1


def test_heuristic_depth_5():
    assert Terminal("ack_count").depth() == 1


def test_heuristic_depth_6():
    assert Unary("neg", Terminal("rate")).depth() == 2


def test_heuristic_depth_7():
    assert Unary("abs", Number(-1.0)).depth() == 2


def test_heuristic_depth_8():
    assert Unary("sqrt", Number(-4.0)).depth() == 2


def test_heuristic_depth_9():
    assert Unary("neg", Unary("sqrt", Terminal("ack_count"))).depth() == 3


def test_heuristic_depth_10():
    assert Binary("plus", Terminal("rate"), Number(1.0)).depth() == 2


def test_heuristic_depth_11():
    assert Binary("sub", Terminal("rate"), Number(1.0)).depth() == 2


def test_heuristic_depth_12():
    assert Binary("mul", Terminal("rate"), Number(2.0)).depth() == 2


def test_heuristic_depth_13():
    assert (
        Binary(
            "plus", Terminal("rate"), Binary("mul", Terminal("rate"), Number(2.0))
        ).depth()
        == 3
    )


def test_heuristic_depth_14():
    assert (
        Binary(
            "max", Unary("neg", Terminal("rate")), Unary("abs", Terminal("ack_count"))
        ).depth()
        == 3
    )
