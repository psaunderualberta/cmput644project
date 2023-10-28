from lark import Lark 
from src.util.constants import HEURISTIC_GRAMMAR
from src.heuristic.transformer import HeuristicTransformer

def parse_heuristic(heuristic):
    # 
    parser = Lark(HEURISTIC_GRAMMAR, start="heuristic", parser="lalr", transformer=HeuristicTransformer())
    return parser.parse(heuristic)

if __name__ == "__main__":
    for h in ["(abs 1.3)", "(+ 5.0 1.0)", "(sqr (max 5.1 10.0))", "(neg (abs rate))"]:
        parsed = parse_heuristic(h)
        print(h)
        print(parsed)
        print(str(parsed))
        print()
