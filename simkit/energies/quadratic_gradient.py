


def quadratic_gradient(x, Q, b):
    """
    Computes a generic quadratic energy gradient.

    Parameters
    ----------
    x : (n, 1) numpy array
        positions of the elastic system
    Q : (n, n) numpy array
        Quadratic matrix
    b : (n, 1) numpy array
        Quadratic vector

    Returns
    -------
    e : float
        Quadratic energy of the system
    """
    e =  Q @ x + b
    return e