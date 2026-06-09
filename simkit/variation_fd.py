"""Finite-difference directional derivative of a Hessian."""


def variation_fd(U0, V, func, epsilon=1e-5):
    """Central-difference directional derivative of a Hessian.

    Approximates ``dH/dx : V`` -- the change in the Hessian ``H_func(x)``
    evaluated at ``x = U0`` along the direction ``V`` -- with a symmetric
    (second-order accurate) finite difference.

    Parameters
    ----------
    U0 : np.ndarray
        Point at which to evaluate the Hessian variation.
    V : np.ndarray
        Direction along which to differentiate, same shape as ``U0``.
    func : callable
        Maps a point ``x`` to its Hessian ``func(x)`` (dense or sparse).
        Must support evaluation at ``U0 +/- epsilon * V``.
    epsilon : float, optional
        Finite-difference step size. Default ``1e-5``.

    Returns
    -------
    dH : np.ndarray or scipy.sparse matrix
        Central-difference estimate
        ``(func(U0 + epsilon * V) - func(U0 - epsilon * V)) / (2 * epsilon)``,
        matching the type and shape returned by ``func``.
    """
    Hm = func(U0 - epsilon * V)
    Hp = func(U0 + epsilon * V)

    dH = (Hp - Hm) / (2 * epsilon)

    return dH
