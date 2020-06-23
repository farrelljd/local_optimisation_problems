from lop.optimise import bfgs
from lop.potentials import CoulombPotentialPolar

from lennard_jones_clusters import find_minima


def question4():
    trials = 100
    d = 2
    for n in range(3,14):
        minima = find_minima(n, d, trials, CoulombPotentialPolar(n, d), method=bfgs)
        print(n, len(minima), min(minima.keys()))


if __name__ == "__main__":
    question4()
