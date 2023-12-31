from lark import Transformer

from src.heuristic.expressions import Binary, Number, Terminal, Unary


class HeuristicTransformer(Transformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def binary(self, args):
        return Binary(args[0].data, args[1], args[2])

    def unary(self, args):
        return Unary(args[0].data, args[1])

    def terminal(self, args):
        return Terminal(args[0].data)

    def decimal(self, args):
        return Number(float(args[0].value))
