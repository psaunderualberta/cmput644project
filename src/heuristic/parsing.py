from lark import Lark

from src.heuristic.transformer import HeuristicTransformer
from src.utility.constants import HEURISTIC_GRAMMAR


def parse_heuristic(heuristic):
    parser = Lark(HEURISTIC_GRAMMAR, start="heuristic")
    return HeuristicTransformer().transform(parser.parse(heuristic))
