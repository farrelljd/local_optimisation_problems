import numpy as np

from .constants import LEVI_CIVITA


def centre_of_mass(pos, masses):
    """

    compute the centre of mass of a system

    :param pos: float array, shape (n, d)
    :param masses: float array, shape (n, )
    :return: float array, shape (d, )
    """
    return np.einsum('i, ia', masses, pos) / masses.sum()


def inertia_tensor(pos, masses):
    """

    compute the inertia tensor of a system

    :param pos: float array, shape (n, d)
    :param masses: float array, shape (n, )
    :return: float array, shape (d, d)
    """
    pos0 = pos - centre_of_mass(pos, masses)
    k = np.einsum('i, ia, ib -> ab', masses, pos0, pos0)
    return np.einsum('age, bde, gd -> ab', LEVI_CIVITA, LEVI_CIVITA, k)


def align(pos, masses=None):
    """

    remove centre of mass &
    align principal axes of inertia with coordinate axes

    if masses is None, all masses == 1

    :param pos: float array, shape (n, d)
    :param masses: float array, shape (n, )
    :return: float array, shape (n, d)
    """
    print(pos)
    if masses is None:
        masses = np.ones_like(pos[:, 0])
    i_tensor = inertia_tensor(pos, masses)
    _, evecs = np.linalg.eigh(i_tensor)
    com = centre_of_mass(pos, masses)
    return np.einsum('ia,ab->ib', pos - com, evecs)
