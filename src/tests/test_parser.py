from src.heuristic.parsing import parse_heuristic
from src.heuristic.transformer import Binary, Unary, Terminal, Number

def test_heuristic_grammar():
    assert parse_heuristic("(abs 1.3)") == Unary("abs", Number(1.3))

def test_heuristic_transformer():
    assert str(parse_heuristic("(abs 1.3)")) == "(abs 1.3)"