from src.utility.util import operators

class Binary:
    def __init__(self, op, left, right):
        (self.op, self.fun) = operators(op)
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"({self.op} {self.left} {self.right})"

    def __eq__(self, other):
        return isinstance(other, Binary) and self.op == other.op and self.left == other.left and self.right == other.right

class Unary:
    def __init__(self, op, right):
        (self.op, self.fun) = operators(op)
        self.right = right
    
    def __repr__(self):
        return f"({self.op} {self.right})"

    def __eq__(self, other):
        return isinstance(other, Unary) and self.op == other.op and self.right == other.right

class Terminal:
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return str(self.data)
    
    def __eq__(self, other):
        return isinstance(other, Terminal) and self.data == other.data

class Number:
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return str(self.value)
    
    def __eq__(self, other):
        return isinstance(other, Number) and self.value == other.value
