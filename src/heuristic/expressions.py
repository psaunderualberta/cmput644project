from src.utility.util import operators
import numpy as np

# Abstract Base Class
class __Heuristic:
    def __repr__(self):
        raise NotImplementedError("This is an abstract class")

    def __eq__(self):
        raise NotImplementedError("This is an abstract class")

    def size(self):
        raise NotImplementedError("This is an abstract class")

    def depth(self):
        raise NotImplementedError("This is an abstract class")

    def execute(self, data):
        raise NotImplementedError("This is an abstract class")
    
    def num_unique_terminals(self, data):
        raise NotImplementedError("This is an abstract class")
    
    def unique_terminals(self, data):
        raise NotImplementedError("This is an abstract class")

class Binary(__Heuristic):
    def __init__(self, op, left, right):
        (self.op, self.fun) = operators(op)
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"({self.op} {self.left} {self.right})"

    def __eq__(self, other):
        return isinstance(other, Binary) and self.op == other.op and self.left == other.left and self.right == other.right

    def size(self):
        return 1 + self.left.size() + self.right.size()
    
    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())

    def execute(self, data):
        return self.fun(self.left.execute(data), self.right.execute(data))

    def num_unique_terminals(self, data):
        return len(self.unique_terminals(data))
    
    def unique_terminals(self, data):
        left_terminals = self.left.unique_terminals(data)
        right_terminals = self.right.unique_terminals(data)
        return left_terminals.union(right_terminals)

class Unary(__Heuristic):
    def __init__(self, op, right):
        (self.op, self.fun) = operators(op)
        self.right = right
    
    def __repr__(self):
        return f"({self.op} {self.right})"

    def __eq__(self, other):
        return isinstance(other, Unary) and self.op == other.op and self.right == other.right
    
    def size(self):
        return 1 + self.right.size()
    
    def depth(self):
        return 1 + self.right.depth()
    
    def execute(self, data):
        return self.fun(self.right.execute(data))
    
    def num_unique_terminals(self, data):
        return self.right.num_unique_terminals(data)
    
    def unique_terminals(self, data):
        return self.right.unique_terminals(data)

class Terminal(__Heuristic):
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return str(self.data)
    
    def __eq__(self, other):
        return isinstance(other, Terminal) and self.data == other.data
    
    def size(self):
        return 1
    
    def depth(self):
        return 1
    
    def execute(self, data):
        return data[self.data].values
    
    def num_unique_terminals(self, data):
        return 1
    
    def unique_terminals(self, data):
        return set([self.data])

class Number(__Heuristic):
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return str(self.value)
    
    def __eq__(self, other):
        return isinstance(other, Number) and self.value == other.value

    def size(self):
        return 1
    
    def depth(self):
        return 1
    
    def execute(self, data):
        return np.full(len(data), self.value)
    
    def num_unique_terminals(self, data):
        return 0
    
    def unique_terminals(self, data):
        return set()