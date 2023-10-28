from lark import Lark 
from util.constants import HEURISTIC_GRAMMAR
from transformer import HeuristicTransformer

def parse_heuristic(heuristic):
    #,  parser="lalr", transformer=HeuristicTransformer()
    parser = Lark(HEURISTIC_GRAMMAR, start="heuristic")
    return parser.parse(heuristic)

if __name__ == "__main__":
    print(parse_heuristic('(abs 1)'))