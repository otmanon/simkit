


def quadratic_energy(x, Q, b):
    """
    Computes a generic quadratic energy.

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
    e = 0.5 * x.T @ Q @ x + b.T @ x
    return e





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