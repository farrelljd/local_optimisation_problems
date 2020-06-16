import sys

import numpy as np

from lop.optimise import *
from lop.potentials import *


def distances_from_projection(r, theta, R):
    """

    For points in polar coordinates (r, theta), find the distance of that point from the corners of an equilateral
    triangle with outradius R and centred at the origin with vertices at theta = (0, 2π/3, 4π/3)

    :param r: radial coordinate
    :type r: ndarray
    :param theta: angular coordinate
    :type theta: ndarray, shape of r
    :param R: outradius of the triangle
    :type R: positive float
    :return: distances
    :rtype: ndarray(*r.shape, 3)

    """
    thetas = np.array([0.0, 2 * np.pi / 3, -2 * np.pi / 3])
    return np.sqrt(R * R + (r * r)[..., None] - 2 * R * r[..., None] * np.cos(theta[..., None] - thetas[None]))


def coords_from_distances(distances):
    """

    From sets of three distances representing the edge lengths of a triangle, find cartesian coordinates for the
    vertices of the triangle in the 2-D.

    :param distances: triangle distances
    :type distances: ndarray(..., 3)
    :return: coordinates of vertices
    :rtype: ndarray(..., 6)

    """
    a, b, c = (distances[..., i] for i in range(3))
    xia = np.zeros((*a.shape, 6), dtype=float)
    xia[..., 2] = a
    cosC = (a * a + b * b - c * c) / (2 * a * b)
    sinC = np.sqrt(1 - cosC * cosC)
    xia[..., -2] = b * cosC
    xia[..., -1] = b * sinC
    return xia


def distances_from_coords(xia):
    """

    From the cartesian coordinates of the vertices of a triangle, find the edge lengths of the triangle.

    :param xia: coordinates of vertices
    :type xia: ndarray(6, )
    :return: lengths
    :rtype: ndarray(3, )

    """
    xia = xia.reshape(3, 2)
    return np.linalg.norm(xia[[0, 0, 1]] - xia[[1, 2, 2]], axis=1)


def make_plot(method=newton_raphson):
    """

    Plot the basins of attraction for a system of 3 Lennard Jones particles using optimisation method 'method'.
    method must be the string name of a method know to scipy.optimize.minimize, i.e., one of
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',
               'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    or a callable compatible with scipy.optimize.minimize.

    The plot and data are saved

    :param method:
    :type method: callable or string
    :return: None

    """
    r_samples = 60
    n = 20
    t_samples = 3 * n + 1
    r = np.linspace(0, 1, r_samples) ** 0.5 * 2
    theta = np.linspace(0, 2 * np.pi, t_samples)
    r, theta = np.meshgrid(r, theta)
    distances = distances_from_projection(r, theta, R=2 ** (1 / 6))
    xias = coords_from_distances(distances)

    n, d = 3, 2
    potential = LennardJonesPotential(n, d)
    from scipy.optimize import minimize

    def minimise(xia, method='Newton-CG', tol=1e-5, **kwargs):
        ret = minimize(fun=potential.get_energy,
                       x0=xia,
                       jac=potential.get_gradient,
                       hess=potential.get_hessian,
                       method=method,
                       tol=tol,
                       **kwargs)
        return ret if ret.success else None

    results = np.zeros_like(theta, dtype=float)

    for i in range(t_samples):
        for j in range(r_samples):
            xia = xias[i, j]
            new = minimise(xia,
                           method=method,
                           tol=1e-5)
            if new is None:
                results[i, j] = -1
                continue
            a, b, c = distances_from_coords(new.x)
            sys.stdout.write(f'({i},{j})\t{new.fun:6.2f} {a:.2f} {b:.2f} {c:.2f}\n')
            sys.stdout.flush()
            results[i, j] = -new.fun

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, subplot_kw={'projection': 'polar'})

    cmap2 = plt.cm.get_cmap("magma", 5)
    norm2 = mpl.colors.Normalize(vmin=-1.0, vmax=3.0)
    cf = ax.pcolormesh(theta, r, results, cmap=cmap2, norm=norm2)
    cb = fig.colorbar(cf)
    cb.set_ticks(np.linspace(-1, 3, 11)[1::2])
    cb.set_ticklabels(["triangular", "linear", "diss. pair", "dissociated", "failure"][::-1])
    ax.set_yticks([])

    ticks = theta[:-1, 0]
    ax.set_xticks([ticks[1] / 2, 2 * np.pi / 3 + ticks[1] / 2, 4 * np.pi / 3 + ticks[1] / 2])
    ax.set_xticklabels((f'{np.degrees(r):.0f}' for r in (0, 2 * np.pi / 3, 4 * np.pi / 3)))

    name = method.__name__ if callable(method) else method
    plt.savefig(f'{name}.png')
    np.save(f'{name}.npy', np.stack([theta, r, results]))

    plt.show()


def main(method):
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',
               'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    methods_dict = {'newton_raphson': newton_raphson, 'bfgs': bfgs, 'steepest_descent': steepest_descent}

    if method in methods:
        make_plot(method)
    elif method in methods_dict:
        make_plot(methods_dict[method])
    else:
        raise ValueError(f'no such method as {method}')


if __name__ == '__main__':
    method = sys.argv[1]
    main(method)
