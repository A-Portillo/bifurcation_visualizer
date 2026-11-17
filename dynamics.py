import ast
import math
import numpy as np
import sympy as sp

x, r = sp.symbols('x r', real=True)


def parse_polynomial(source):
    """Parse arithmetic without eval; accept polynomials in x and r only."""
    if len(source) > 400:
        raise ValueError('Use at most 400 characters.')
    try:
        tree = ast.parse(source.replace('^', '**'), mode='eval')
    except SyntaxError as exc:
        raise ValueError('Invalid expression. Use explicit multiplication, e.g. r*x - x^3.') from exc
    if sum(1 for _ in ast.walk(tree)) > 150:
        raise ValueError('Expression is too complex.')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 1e12:
                raise ValueError('Constants must be finite and at most 10^12 in magnitude.')
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in ('x', 'r'):
            return {'x': x, 'r': r}[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div) and b.is_number and b != 0:
                return a / b
            if isinstance(node.op, ast.Pow) and b.is_Integer and 0 <= b <= 20:
                if sp.Poly(a, x, r).total_degree() * int(b) > 20:
                    raise ValueError('Supported total degree is at most 20.')
                if a.is_number and abs(a) > 1e12:
                    raise ValueError('Numeric power base is too large.')
                return a ** b
        raise ValueError('Use only x, r, numbers, +, -, *, / by constants, and nonnegative integer powers. t is the independent variable.')

    expression = visit(tree.body)
    poly = sp.Poly(expression, x, r)
    if poly.total_degree() > 20 or len(poly.terms()) > 150:
        raise ValueError('Supported total degree is at most 20, with at most 150 terms.')
    return poly.as_expr()


def coefficients(expression, parameter):
    return np.array([float(c) for c in sp.Poly(expression.subs(r, float(parameter)), x).all_coeffs()])


def equilibria(expression, parameter):
    coeff = np.trim_zeros(coefficients(expression, parameter), 'f')
    if not len(coeff):
        return None  # Every x is an equilibrium.
    if len(coeff) == 1:
        return []
    roots = np.polynomial.polynomial.polyroots(coeff[::-1])
    real = sorted(float(z.real) for z in roots if abs(z.imag) < 1e-6 * (1 + abs(z.real)))
    unique = []
    for value in real:
        if not unique or abs(value - unique[-1]) > 1e-5 * (1 + abs(value)):
            unique.append(value)
    return unique


def stability(expression, parameter, root):
    """First nonzero Taylor term also classifies nonhyperbolic roots."""
    poly = sp.Poly(expression.subs(r, float(parameter)), x)
    # Translation is more reliable than sampling beside closely spaced roots.
    coeff = np.array([float(c) for c in poly.all_coeffs()])
    scale = max(1.0, np.max(np.abs(coeff)))
    for order in range(1, poly.degree() + 1):
        value = float(sp.diff(poly.as_expr(), x, order).subs(x, root)) / math.factorial(order)
        if abs(value) > 1e-7 * scale * (1 + abs(root)) ** max(0, poly.degree() - order):
            if order % 2 == 0:
                return 'Semistable'
            return 'Stable' if value < 0 else 'Unstable'
    return 'Nonhyperbolic'


def critical_parameters(expression, lower, upper):
    """Finite candidates: repeated roots and changes in polynomial degree."""
    poly = sp.Poly(expression, x)
    if poly.degree() < 1 or sp.Poly(expression, x, r).total_degree() > 8:
        return []
    candidates = []
    for condition in (sp.discriminant(expression, x), poly.LC()):
        if condition == 0 or not condition.has(r):
            continue
        values = np.polynomial.polynomial.polyroots(np.array([float(c) for c in sp.Poly(condition, r).all_coeffs()][::-1]))
        candidates.extend(float(z.real) for z in values if abs(z.imag) < 1e-6 and lower <= z.real <= upper)
    return sorted(set(round(value, 10) for value in candidates))
