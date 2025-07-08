


def quadratic_hessian(Q):
    """
    Computes a generic quadratic energy hessian. Sorta redundant but I like having a standard quadratic form.

    Parameters
    ----------
    Q : (n, n) numpy array
        Quadratic matrix

    Returns
    -------
    Q : float
        Quadratic energy of the system
    """
    return Q