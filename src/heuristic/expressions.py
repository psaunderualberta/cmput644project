from src.utility.util import operators


# Abstract Base Class
class __Heuristic:
    __abc_err_msg = "This is an abstract class"

    def __repr__(self):
        raise NotImplementedError(self.__abc_err_msg)

    def __eq__(self, _):
        raise NotImplementedError(self.__abc_err_msg)

    def __ne__(self, _):
        raise NotImplementedError(self.__abc_err_msg)

    def size(self):
        raise NotImplementedError(self.__abc_err_msg)

    def num_unique_terminals(self):
        raise NotImplementedError(self.__abc_err_msg)

    def unique_terminals(self):
        raise NotImplementedError(self.__abc_err_msg)

    def depth(self):
        raise NotImplementedError(self.__abc_err_msg)

    def execute(self, _):
        raise NotImplementedError(self.__abc_err_msg)

    def __hash__(self):
        raise NotImplementedError(self.__abc_err_msg)


class Binary(__Heuristic):
    def __init__(self, op, left, right):
        (self.op, self.fun) = operators(op)
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.op} {self.left} {self.right})"

    def __eq__(self, other):
        return (
            isinstance(other, Binary)
            and self.op == other.op
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self):
        return hash(repr(self))

    def size(self):
        return 1 + self.left.size() + self.right.size()

    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())

    def execute(self, data):
        return self.fun(self.left.execute(data), self.right.execute(data))

    def num_unique_terminals(self):
        return len(self.unique_terminals())

    def unique_terminals(self):
        left_terminals = self.left.unique_terminals()
        right_terminals = self.right.unique_terminals()
        return left_terminals.union(right_terminals)


class Unary(__Heuristic):
    def __init__(self, op, right):
        (self.op, self.fun) = operators(op)
        self.right = right

    def __repr__(self):
        return f"({self.op} {self.right})"

    def __eq__(self, other):
        return (
            isinstance(other, Unary)
            and self.op == other.op
            and self.right == other.right
        )

    def __hash__(self):
        return hash(repr(self))

    def size(self):
        return 1 + self.right.size()

    def depth(self):
        return 1 + self.right.depth()

    def execute(self, data):
        return self.fun(self.right.execute(data))

    def num_unique_terminals(self):
        return self.right.num_unique_terminals()

    def unique_terminals(self):
        return self.right.unique_terminals()


class Terminal(__Heuristic):
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return str(self.data)

    def __eq__(self, other):
        return isinstance(other, Terminal) and self.data == other.data

    def __hash__(self):
        return hash(repr(self))

    def size(self):
        return 1

    def depth(self):
        return 1

    def execute(self, data):
        return data[self.data]

    def num_unique_terminals(self):
        return 1

    def unique_terminals(self):
        return set([self.data])


class Number(__Heuristic):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)

    def __eq__(self, other):
        return isinstance(other, Number) and self.value == other.value

    def __hash__(self):
        return hash(repr(self))

    def size(self):
        return 1

    def depth(self):
        return 1

    def execute(self, data):
        return data.iloc[:, 0] * 0 + self.value

    def num_unique_terminals(self):
        return 0

    def unique_terminals(self):
        return set()
