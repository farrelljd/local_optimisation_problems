import numpy as np

from lop.optimise import steepest_descent
from lop.potentials import LennardJonesPotential


def question1():
    n, d = 5, 3
    potential = LennardJonesPotential(n, d)

    xia = np.loadtxt('data/lj5.dat')
    v = potential.get_energy(xia)
    print(potential.get_gradient(xia))
    print(v)


def question2():
    n, d = 5, 3
    potential = LennardJonesPotential(n, d)

    xia = np.loadtxt('data/lj5.dat')

    xia += 0.01 * np.random.rand(xia.size).reshape(xia.shape)
    xia = xia.flatten()

    ret = steepest_descent(fun=potential.get_energy,
                           args=(),
                           x0=xia,
                           jac=potential.get_gradient,
                           tol=1e-7)
    v = potential.get_energy(ret.x)
    print(v)


def find_minima(n, d, trials, potential, method):
    def minimise(xia):
        ret = method(fun=potential.get_energy,
                     args=(),
                     x0=xia,
                     jac=potential.get_gradient,
                     tol=1e-5)
        return ret

    minima = {}

    for trial in range(trials):
        xia = np.random.rand(n * d)
        ret = minimise(xia)
        rounded = np.round(ret.fun, 5)
        if rounded not in minima:
            minima[rounded] = 1
        else:
            minima[rounded] += 1
    return minima


def question3():
    d = 3
    trials = 10
    for n in range(3, 14):
        minima = find_minima(n, d, trials, LennardJonesPotential(n, d), method=steepest_descent)
        print(n, len(minima), min(minima.keys()))


if __name__ == '__main__':
    # question1()
    # question2()
    question3()
