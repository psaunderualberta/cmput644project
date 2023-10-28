from lark import Transformer

class HeuristicTransformer(Transformer):
    class Binary:
        def __init__(self, op, left, right):
            self.op = op
            self.left = left
            self.right = right

    class Unary:
        def __init__(self, op, right):
            self.op = op
            self.right = right
    
    def binary(self, args):
        return self.Binary(args[0], args[1], args[2])

    def unary(self, args):
        return self.Unary(args[0], args[1])
    