from numpy import eye
from numpy.linalg import norm, pinv
from scipy.optimize import OptimizeResult

from .bfgs_update import bfgs_update_hessian
from .line_search import line_search


def bfgs(fun, x0, args, jac, tol=1e-5, **kwargs):
    """
    Using the Broyden-Fletcher-Goldfarb-Shanno method,
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
    x = x0.copy()
    binv = eye(x.size)
    gx = jac(x)
    modg = norm(gx)

    iteration = 0
    while modg > tol:
        iteration += 1
        p = -binv @ gx
        alpha = line_search(fun, x, p, gx)
        x = x + alpha * p

        s = alpha * p
        gx_new = jac(x)
        binv = bfgs_update_hessian(binv, gx_new - gx, s)
        gx = gx_new
        modg = norm(gx)

    return OptimizeResult({'x': x,
                           'success': True,
                           'status': 0,
                           'message': None,
                           'fun': fun(x),
                           'jac': jac(x),
                           'hess_inv': binv,
                           'nit': iteration,
                           })
