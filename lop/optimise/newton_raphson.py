from math import inf

from numpy.linalg import norm, pinv
from scipy.optimize import OptimizeResult


def newton_raphson(fun, x0, args, jac, hess, tol=1e-5, gtol=1e-5, **kwargs):
    """
    Using the Newton-Raphson method,
    find a minimum of a function f with first derivative g
    and Hessian h in the region of the point x0.

    :param fun: the objective function
    :type fun: callable
    :param x0: the initial coordinates
    :type x0: numpy.ndarray
    :param args: additional arguments for fun
    :type args: tuple
    :param jac: the first derivative of the objective function
    :type jac: callable
    :param hess: the second derivative of the objective function
    :type hess: callable
    :param tol: the convergence criterion for the norm of the update
    :type tol: float
    :param gtol: the convergence criterion for the norm of the gradient
    :type gtol: float
    :return: results of the optimisation
    :rtype: scipy.optimize.OptimizeResult

    """
    dx = inf
    x = x0.copy()

    iteration = 0
    while norm(dx) > tol and norm(jac(x)) > gtol:
        iteration += 1
        dx = pinv(hess(x), hermitian=True).dot(jac(x))  # pinv -> pseudo-inverse

        x -= dx

    return OptimizeResult({'x': x,
                           'success': True,
                           'status': 0,
                           'message': None,
                           'fun': fun(x),
                           'jac': jac(x),
                           'hess': hess(x),
                           'nit': iteration,
                           })
