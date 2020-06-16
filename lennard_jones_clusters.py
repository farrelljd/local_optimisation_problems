import numpy as np

from lop.potentials import LennardJonesPotential
from lop.optimise import steepest_descent


def question1():
    n, d = 5, 3
    potential = LennardJonesPotential(n, d)

    xia = np.loadtxt('data/lj5.dat')
    v = potential.get_energy(xia)
    print(v)


def question2():
    n, d = 5, 3
    potential = LennardJonesPotential(n, d)

    xia = np.loadtxt('data/lj5.dat')

    xia += 0.01*np.random.rand(xia.size).reshape(xia.shape)
    xia = xia.flatten()

    ret = steepest_descent(fun=potential.get_energy,
                           args=(),
                           x0=xia,
                           jac=potential.get_gradient,
                           tol=1e-7)
    v = potential.get_energy(ret.x)
    print(v)


if __name__ == '__main__':
    question1()
    question2()
