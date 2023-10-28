from src.heuristic.expressions import Binary, Unary, Terminal, Number

def test_heuristic_size_1():
    assert Number(5.3).size() == 1

def test_heuristic_size_2():
    assert Number(-1.0).size() == 1

def test_heuristic_size_3():
    assert Terminal("rate").size() == 1

def test_heuristic_size_4():
    assert Terminal("syn_flag_number").size() == 1

def test_heuristic_size_5():
    assert Terminal("ack_count").size() == 1

def test_heuristic_size_6():
    assert Unary("neg", Terminal("rate")).size() == 2

def test_heuristic_size_7():
    assert Unary("abs", Number(-1.0)).size() == 2

def test_heuristic_size_8():
    assert Unary("sqrt", Number(-4.0)).size() == 2

def test_heuristic_size_9():
    assert Unary("neg", Unary("sqrt", Terminal("ack_count"))).size() == 3
    
def test_heuristic_size_10():
    assert Binary("plus", Terminal("rate"), Number(1.0)).size() == 3

def test_heuristic_size_11():
    assert Binary("sub", Terminal("rate"), Number(1.0)).size() == 3

def test_heuristic_size_12():
    assert Binary("mul", Terminal("rate"), Number(2.0)).size() == 3

def test_heuristic_size_13():
    assert Binary("plus", Terminal("rate"), Binary("mul", Terminal("rate"), Number(2.0))).size() == 5

def test_heuristic_size_14():
    assert Binary("max", Unary("neg", Terminal("rate")), Unary("abs", Terminal("ack_count"))).size() == 5
