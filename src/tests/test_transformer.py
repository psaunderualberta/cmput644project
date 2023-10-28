from src.heuristic.parsing import parse_heuristic

# Numbers
def test_transformer_number_1():
    assert str(parse_heuristic("1.0")) == "1.0"

def test_transformer_number_2():
    assert str(parse_heuristic("-5.3")) == "-5.3"

# Terminals
def test_transformer_terminal_1():
    assert str(parse_heuristic("rate")) == "rate"

def test_transformer_terminal_2():
    assert str(parse_heuristic("syn_flag_number")) == "syn_flag_number"

# Unaries
def test_transformer_unary_1():
    assert str(parse_heuristic("(abs 1.3)")) == "(abs 1.3)"

def test_transformer_unary_2():
    assert str(parse_heuristic("(neg rate)")) == "(neg rate)"

def test_transformer_unary_3():
    assert str(parse_heuristic("(sqrt 1.3)")) == "(sqrt 1.3)"

def test_transformer_unary_4():
    assert str(parse_heuristic("(sqr 1.3)")) == "(sqr 1.3)"

# Binaries
def test_transformer_binary_1():
    assert str(parse_heuristic("(+ 1.3 1.0)")) == "(+ 1.3 1.0)"

def test_transformer_binary_2():
    assert str(parse_heuristic("(- 1.3 1.0)")) == "(- 1.3 1.0)"

def test_transformer_binary_3():
    assert str(parse_heuristic("(* 1.3 rate)")) == "(* 1.3 rate)"

# Depth > 1
def test_transformer_deep_1():
    assert str(parse_heuristic("(+ 1.3 (* 1.0 rate))")) == "(+ 1.3 (* 1.0 rate))"

def test_transformer_deep_2():
    assert str(parse_heuristic("(max (neg rate) (abs ack_count))")) == "(max (neg rate) (abs ack_count))"