from math import inf
from numpy.linalg import norm
from .line_search import line_search
from scipy.optimize import OptimizeResult


def steepest_descent(fun, x0, args, jac, tol=1e-5, **kwargs):
    """
    Using the steepest descent method,
    find a minimum of a function f with first derivative g
    in the region of the point x0.

    :param fun: the objective function
    :type fun: callable
    :param x0: the initial coordinates
    :type x0: numpy.ndarray
    :param args: additional arguments for fun
    :type args: tuple
    :param jac: the first derivative of the objective function
    :type jac: callable
    :param tol: the convergence criterion
    :type tol: float
    :return: results of the optimisation
    :rtype: scipy.optimize.OptimizeResult
    """

    modg = inf
    x = x0.copy()

    while modg > tol:
        gx = jac(x)
        modg = norm(gx)
        p = -gx / modg
        alpha = line_search(fun, x, p, gx)
        x = x + alpha * p

    return OptimizeResult({'x': x,
                           'success': True,
                           'status': 0,
                           'message': None,
                           'fun': fun(x),
                           'jac': jac(x),
                           })
