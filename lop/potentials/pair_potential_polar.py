import numpy as np


class PairPotentialPolar:
    """

    Same as PairPotential, but on the surface of a unit sphere in spherical polar coordinates.

    """

    def __init__(self, n, d, shift=True):
        self.shape = n, d
        if d != 2:
            raise ValueError(f'PairPotentialPolar dimension must be 2, not {d}')
        self.shift = shift

    def get_distances(self, sia):
        """

        Compute arrays of derivatives of distances and distances

        :param sia: particle position vectors (flattened)
        :type sia: ndarray(n*2, )
        :return: arrays of derivatives of distances and distances
        :rtype:  ndarray(2, n, n), ndarray(n, n)
        """
        n, d = self.shape
        theta, phi = sia.reshape(n, d).T
        st, sp, ct, cp = np.sin(theta), np.sin(phi), np.cos(theta), np.cos(phi)

        dij = np.sqrt(2 - 2 * (st[:, None] * st[None, :] * np.cos(phi[:, None] - phi[None, :]) + ct[:, None] * ct[None, :]))
        dij = np.ma.array(dij)
        dij[np.diag_indices(n)] = np.ma.masked

        ddij = np.stack(
            [-2 * (ct[:, None] * st[None, :] * np.cos(phi[:, None] - phi[None, :]) - st[:, None] * ct[None, :]),
             2 * st[:, None] * st[None, :] * np.sin(phi[:, None] - phi[None, :])])

        return ddij, dij

    def get_energy(self, sia):
        _, dij = self.get_distances(sia)
        return self.pair_energy(dij).sum() / 2

    def get_gradient(self, sia):
        ddij, dij = self.get_distances(sia)
        gij = self.pair_gradient(dij)
        return np.einsum('ij, aij, ij->ia', gij, ddij, 1 / 2 / dij).flatten()

    def pair_energy(self, d):
        raise NotImplementedError

    def pair_gradient(self, d):
        raise NotImplementedError

    def pair_hessian(self, d):
        raise NotImplementedError
