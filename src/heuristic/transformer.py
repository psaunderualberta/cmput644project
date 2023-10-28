from lark import Transformer

class Binary:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"({self.op} {self.left} {self.right})"

class Unary:
    def __init__(self, op, right):
        self.op = op
        self.right = right
    
    def __repr__(self):
        return f"({self.op} {self.right})"

class Terminal:
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return str(self.data)
    
class Number:
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return str(self.value)
    

class HeuristicTransformer(Transformer):    
    def binary(self, args):
        return Binary(args[0].data, args[1], args[2])

    def unary(self, args):
        return Unary(args[0].data, args[1])
    
    def terminal(self, args):
        return Terminal(args[0].data)

    def decimal(self, args):
        return Number(float(args[0].value))
