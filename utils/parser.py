import numpy as np
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,  # it looks for cases where two things are side-by-side without an operator and inserts a *.
    implicit_application,  # it allows users to call functions without using parentheses if the context is clear.
    function_exponentiation,  # cos**2(x)
)

transformations = (
    standard_transformations  # already a tuple of multiple transformations functions
    + (implicit_multiplication_application,)
    + (implicit_application,)
    + (function_exponentiation,)
)


# parse the string input passed by user with numpy functions
def parse_func(expr_str: str):
    expr_str = expr_str.strip().lower()
    x = symbols("x")
    expr = parse_expr(expr_str, transformations=transformations)
    f = lambdify(x, expr, "numpy")
    f_prime = lambdify(x, expr.diff(x), "numpy")  # for newton

    return f, f_prime
