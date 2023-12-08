from sympy import *

init_printing() 
import numpy as np
import pandas as pd

from src.heuristic.expressions import Binary
from src.heuristic.expressions import Number as HeuristicNumber
from src.heuristic.expressions import Terminal, Unary
from src.heuristic.parsing import parse_heuristic
from src.utility.constants import *


class Simplifier():
    def __init__(self, expressions, **kwargs):
        self.cache = dict()

        self.df = pd.DataFrame({
            "expressions": expressions,
            "symbolic": [self.symbolic(h, False) for h in expressions],
            "simplified": [self.symbolic(h, True) for h in expressions]
        })
        self.df = self.df.assign(**kwargs)

    def __simplify_binary(self, binary, evaluate):
        left_simplified = self.symbolic(binary.left, evaluate)
        right_simplified = self.symbolic(binary.right, evaluate)

        return {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: x / y,
            "min": lambda x, y: Min(x, y, evaluate=evaluate),
            "max": lambda x, y: Max(x, y, evaluate=evaluate),
        }[binary.op](left_simplified, right_simplified)
    
    def __simplify_unary(self, heuristic, evaluate):
        right_simplified = self.symbolic(heuristic.right, evaluate)

        return {
            "abs": lambda x: Abs(x, evaluate=evaluate),
            "neg": lambda x: -x,
            "sqr": lambda x: Pow(x, 2, evaluate=evaluate),
            "sqrt": lambda x: sqrt(x, evaluate=evaluate)
        }[heuristic.op](right_simplified)

    def symbolic(self, heuristic, evaluate=False):
        key = (evaluate, str(heuristic))
        if key in self.cache:
            return self.cache[key]

        if isinstance(heuristic, str):
            heuristic = parse_heuristic(heuristic)

        if isinstance(heuristic, Binary):
            simplified = self.__simplify_binary(heuristic, evaluate)
        elif isinstance(heuristic, Unary):
            simplified = self.__simplify_unary(heuristic, evaluate)
        elif isinstance(heuristic, Terminal):
            name = "\\text\{" + heuristic.data.replace("_", " ") + "\}"
            simplified = Symbol(name)
        elif isinstance(heuristic, HeuristicNumber):
            simplified = Number(heuristic.value)
        else:
            raise TypeError(f"Cannot convert '{heuristic}' to symbolic")
            
        
        self.cache[key] = simplified
        return simplified
        
    def equivalent(self, h1, h2):
        h1_symbolic = self.symbolic(h1)
        h2_symbolic = self.symbolic(h2)

        key = tuple(sorted((str(h1_symbolic), str(h2_symbolic))))
        if key in self.equivalent_cache:
            return self.equivalent_cache[key]

        h1_simplified = simplify(h1_symbolic)
        h2_simplified = simplify(h2_symbolic)
        self.equivalent_cache[key] = simplify(h1_simplified - h2_simplified) == 0

        return self.equivalent_cache[key]

    def intersect(self, arr1, arr2):
        intersection = set()
        for heuristic in arr1:
            if any(self.equivalent(heuristic, h) for h in arr2):
                intersection.add(heuristic)
        
        return list(intersection)
    
    def get_equivalencies(self):
        equiv_idxs = {i: set() for i in range(len(self.df))}
        to_remove = []

        for i, h1 in enumerate(self.df["simplified"]):
            for j, h2 in enumerate(self.df["simplified"].loc[i+1:]):
                if hash(h1) == hash(h2):  # simplify(h1 - h2) == 0
                    equiv_idxs[i].add(j + i + 1)
                    to_remove.append(j + i + 1)
        
        sym = self.df["symbolic"]
        equiv = {latex(sym.loc[key]): [latex(sym.loc[j]) for j in vals] for key, vals in equiv_idxs.items() if key not in to_remove}
        
        return equiv
    
    def get_df(self):
        return self.df.copy()

if __name__ == "__main__":
    s = Simplifier()
    print(s.symbolic("(neg (+ rate 1.0))"))