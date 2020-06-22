from .pair_potential_polar import PairPotentialPolar


class CoulombPotentialPolar(PairPotentialPolar):
    """
    1-D Coulomb energy plus first and second derivatives
    """

    def pair_energy(self, d):
        return 1/d

    def pair_gradient(self, d):
        return -1/d/d

    def pair_hessian(self, d):
        return 2/d/d/d
