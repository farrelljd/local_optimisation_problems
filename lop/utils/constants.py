def levi_civita(dtype=float):
    """

    Build the 3-D Levi--Civita matrix

    :param dtype: data type of the matrix
    :return: Levi--Civita matrix
    :rtype: ndarray(3, 3, 3)
    """
    from numpy import zeros
    e_tensor = zeros([3, 3, 3], dtype=dtype)
    for i in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        e_tensor.itemset(i, 1)
        e_tensor.itemset(i[::-1], -1)
    e_tensor.setflags(write=False)
    return e_tensor


LEVI_CIVITA = levi_civita(int)
