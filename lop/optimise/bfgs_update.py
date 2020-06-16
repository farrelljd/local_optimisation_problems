from numpy import newaxis


def bfgs_update_hessian(binv, y, s):
    """
    Update the approximate inverse Hessian matrix in the BFGS scheme
    using the Sherman-Morrison formula

    :param binv: the approximate inverse Hessian at step i
    :type binv: numpy.ndarray, shape [n,n]
    :param y: the difference between gradient at step i+1 and step i
    :type y: numpy.ndarray, shape [n,]
    :param s: the BFGS coordinate update at step i
    :type s: numpy.ndarray, shape [n,]
    :return: the approximate inverse Hessian at step i+1

    """
    y, yT = y[:, newaxis], y[newaxis, :]
    s, sT = s[:, newaxis], s[newaxis, :]

    return (binv +
            (sT @ y + yT @ binv @ y) * (s @ sT) / (sT @ y) ** 2 -
            (binv @ y @ sT + s @ yT @ binv) / (sT @ y))
