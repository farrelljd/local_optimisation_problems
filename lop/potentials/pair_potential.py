import numpy as np


class PairPotential:
    """

    calculate energies, gradients, and hessians for isotropic, pairwise potentials
    number of particles and dimensions variable

    overload pair_energy, pair_gradient, and pair_hessian

    """

    def __init__(self, n, d, shift=True):
        self.shape = n, d
        self.shift = shift

    def get_distances(self, xia):
        """

        Compute arrays of interparticle vectors and distances

        :param xia: particle position vectors (flattened)
        :type xia: ndarray(n*d, )
        :return: arrays of interparticle vectors and distances
        :rtype:  ndarray(n, n-1, d), ndarray(n, n-1)
        """
        n, d = self.shape
        xia = xia.reshape(n, d)
        dija = xia[:, None] - xia[None, :]
        dij = np.linalg.norm(dija, axis=2)
        dij = np.ma.masked_equal(dij, value=0.0, copy=True)
        return dija, dij

    def get_energy(self, xia):
        _, dij = self.get_distances(xia)
        return self.pair_energy(dij).sum() / 2

    def get_gradient(self, xia):
        dija, dij = self.get_distances(xia)
        ngij = self.pair_gradient(dij) / dij
        return np.einsum('ij, ija->ia', ngij, dija).flatten()

    def get_hessian(self, xia):
        n, d = self.shape
        dija, dij = self.get_distances(xia)
        ngij = self.pair_gradient(dij) / dij
        hij = self.pair_hessian(dij) - ngij
        ndiajb = np.einsum('ija, ijb, ij->iajb', dija, dija, 1 / dij / dij)
        h = -np.einsum('iajb, ij->iajb', ndiajb, hij)  # iajb
        h[:, range(d), :, range(d)] -= ngij  # iaja
        h[range(n), :, range(n), :] -= np.einsum('iajb->iab', h)  # iaia, iaib
        h = h.reshape(n * d, n * d)
        if self.shift:
            h += self.shift_eigenvalues()
        return h

    def shift_eigenvalues(self, value=100):
        """

        shift the eigenvalues of the Hessian that correspond to pure translations

        :return: the shift
        :rtype: ndarray(n*d, n*d)
        """
        n, d = self.shape
        s = np.zeros((n * d, n * d), dtype=float)
        for i in range(d):
            s[i::d, i::d] += 1
        return value * s

    def pair_energy(self, d):
        raise NotImplementedError

    def pair_gradient(self, d):
        raise NotImplementedError

    def pair_hessian(self, d):
        raise NotImplementedError
