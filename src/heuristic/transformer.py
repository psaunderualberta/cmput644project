from lark import Transformer
from src.heuristic.expressions import Binary, Unary, Terminal, Number


class HeuristicTransformer(Transformer):

    def __init__(self, *args, **kwargs):
        if "dask" in kwargs:
            self.dask = True
            del kwargs["dask"]
        else:
            self.dask = False
        super().__init__(*args, **kwargs)

    def binary(self, args):
        return Binary(args[0].data, args[1], args[2], dask=self.dask)

    def unary(self, args):
        return Unary(args[0].data, args[1], dask=self.dask)

    def terminal(self, args):
        return Terminal(args[0].data, dask=self.dask)

    def decimal(self, args):
        return Number(float(args[0].value), dask=self.dask)
