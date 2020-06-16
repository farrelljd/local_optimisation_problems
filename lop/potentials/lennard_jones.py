from .pair_potential import PairPotential


class LennardJonesPotential(PairPotential):
    """
    1-D Lennard Jones energy plus first and second derivatives
    """

    def pair_energy(self, d):
        d6 = 1 / (d * d)
        d6 *= d6 * d6
        return 4 * d6 * (d6 - 1)

    def pair_gradient(self, d):
        d6 = 1 / (d * d)
        d6 *= d6 * d6
        return 24 * d6 * (1 - 2 * d6) / d

    def pair_hessian(self, d):
        d6 = 1 / (d * d)
        d6 *= d6 * d6
        return 24 * d6 * (26 * d6 - 7) / d / d
