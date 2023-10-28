from src.heuristic.parsing import parse_heuristic
from src.heuristic.transformer import Binary, Unary, Terminal, Number

# Numbers
def test_heuristic_parsing_number_1():
    assert parse_heuristic("1.0") == Number(1.0)

def test_heuristic_parsing_number_2():
    assert parse_heuristic("-5.3") == Number(-5.3)

# Terminals
def test_heuristic_parsing_terminal_1():
    assert parse_heuristic("rate") == Terminal("rate")

def test_heuristic_parsing_terminal_2():
    assert parse_heuristic("syn_flag_number") == Terminal("syn_flag_number")

# Unaries
def test_heuristic_parsing_unary_1():
    assert parse_heuristic("(abs 1.3)") == Unary("abs", Number(1.3))

def test_heuristic_parsing_unary_2():
    assert parse_heuristic("(neg rate)") == Unary("neg", Terminal("rate"))

def test_heuristic_parsing_unary_3():
    assert parse_heuristic("(sqrt 1.3)") == Unary("sqrt", Number(1.3))

def test_heuristic_parsing_unary_4():
    assert parse_heuristic("(sqr 1.3)") == Unary("sqr", Number(1.3))

# Binaries
def test_heuristic_parsing_binary_1():
    assert parse_heuristic("(+ 1.3 1.0)") == Binary("plus", Number(1.3), Number(1.0))

def test_heuristic_parsing_binary_2():
    assert parse_heuristic("(- 1.3 1.0)") == Binary("sub", Number(1.3), Number(1.0))

def test_heuristic_parsing_binary_3():
    assert parse_heuristic("(* 1.3 rate)") == Binary("mul", Number(1.3), Terminal("rate"))

# Depth > 1
def test_heuristic_parsing_deep_1():
    assert parse_heuristic("(+ 1.3 (* 1.0 rate))") == Binary("plus", Number(1.3), Binary("mul", Number(1.0), Terminal("rate")))

def test_heuristic_parsing_deep_2():
    assert parse_heuristic("(max (neg rate) (abs ack_count))") == Binary("max", Unary("neg", Terminal("rate")), Unary("abs", Terminal("ack_count")))