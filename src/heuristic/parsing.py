from lark import Lark
from src.utility.constants import HEURISTIC_GRAMMAR
from src.heuristic.transformer import HeuristicTransformer


def parse_heuristic(heuristic):
    parser = Lark(
        HEURISTIC_GRAMMAR,
        start="heuristic",
        parser="lalr",
        transformer=HeuristicTransformer(),
    )
    return parser.parse(heuristic)
