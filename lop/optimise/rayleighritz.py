import numpy as np
from .bfgs import bfgs


def rayleigh_ritz(h, x, tol=1e-3):
    """
    Finds the smallest eigenvalue and associated eigenvector
    of the Hessian matrix at point x by minimising the Rayleigh-Ritz ratio,

        λ(v) = (v.T . H . v) / (v.T . v)^2

    :param h: function to get the Hessian
    :type h: callable
    :param x: point at which to evaluate the Hessian
    :type x: numpy.ndarray
    :param tol: convergence tolerance for the gradient of λ
    :type tol: float
    :return: the smallest eigenvalue and associated (normalised) eigenvector
    :rtype: (float, numpy.ndarray,)

    """
    hessian = h(x)
    v0 = np.random.rand(hessian.shape[0])

    def fun(v):
        v_ = v[:, np.newaxis]
        return ((v_.T @ hessian @ v_) / (v_.T @ v_))[0, 0]

    def jac(v):
        v_ = v[:, np.newaxis]
        left = (v_.T @ hessian @ v_)
        left_prime = hessian @ v_ + hessian.T @ v_
        right = (v_.T @ v_)
        right_prime = 2 * v_
        return ((left_prime * right - left * right_prime) / (right * right)).flatten()

    vmin = bfgs(fun, v0, None, jac, tol).x

    return fun(vmin), vmin / np.linalg.norm(vmin)
