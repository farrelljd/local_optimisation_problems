import numpy as np
from numpy.linalg import norm, pinv
from scipy.optimize import OptimizeResult

from .bfgs_update import bfgs_update_hessian
from .line_search import line_search
from .rayleighritz import rayleigh_ritz


def project_out(u, v):
    return u - (u @ v) * v


def hybrid_eigenvector_following(fun, x0, args, jac, hess, tol=1e-5, tolev=1e-6, **kwargs):
    """
    Converge to a first-order saddle of function f
    using hybrid-eigenvector following.

    The smallest eigenvalue and eigenvector of the Hessian are determined
    from the Rayleigh--Ritz ratio, and the uphill step is taken via a line search.

    Minimisation in the subspace orthogonal to the smallest eigenvector is done
    with the BFGS method.

    :param fun: the function
    :type fun: callable
    :param x0: initial coordinates
    :type x0: numpy.ndarray
    :param args: additional arguments for fun
    :type args: tuple
    :param jac: function to evaluate the gradient of f
    :type jac: callable
    :param hess: function to evaluate the Hessian of h
    :type hess: callable
    :param tol: tolerance in the rms gradient
    :type tol: float
    :param tolev: tolerance in the Rayleigh-Ritz minimisation
    :type tolev: float
    :return: results of the optimisation
    :rtype: scipy.optimize.OptimizeResult

    """
    x = x0.copy()
    gx = jac(x)
    modg = norm(gx)
    binv = np.eye(x.size)

    def negativef(x):  # function to evaluate (-f(x)) for use in uphill search
        return -fun(x)

    while modg > tol:
        eval_, evec = rayleigh_ritz(hess, x, tolev)  # find the negative e'nvalue

        if eval_ > 0:
            raise ValueError(f"smallest eigenvalue is positive {eval_:.2e}")
        gx_ = -(gx @ evec) * evec  # gradient along e'nvector

        if norm(gx_) > tol:
            alpha = line_search(negativef, x, evec, gx_, max_alpha=0.1)
            x = x + alpha * evec

        p = -binv @ gx
        gx = project_out(gx, evec)  # project e'nvector out of gradient,
        p = project_out(p, evec)  # search direction
        p /= norm(p)

        if norm(gx) > tol:
            alpha = line_search(fun, x, p, gx, max_alpha=1.0)
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
                           'hess': hess(x)
                           })
