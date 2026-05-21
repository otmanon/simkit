# import numpy as np


# # def friction_interpolant(y, epsilon_v, h):
# #     f = y.copy()**3 
# #     return f

# # def friction_interpolant_gradient(y, epsilon_v, h):
# #     g =  3* y.copy()**2 
# #     return g

# # def friction_interpolant_hessian(y, epsilon_v, h):
# #     h =  6* y.copy() 
# #     return h


# # def friction_interpolant(y, epsilon_v, h):
# #     threshold = h * epsilon_v
# #     f = - y**3/(3 *threshold**2) +  y**2/threshold

# #     return f

# # def friction_interpolant_gradient(y, epsilon_v, h):
# #     threshold = h * epsilon_v
# #     g =  - y**2/(threshold**2) + 2 * y/threshold
# #     return g

# # def friction_interpolant_hessian(y, epsilon_v, h):
# #     threshold = h * epsilon_v
# #     h = - 2 * y / (threshold**2) + 2 * np.ones(y.shape)/ threshold
# #     return h



# def sticking_friction_interpolant(y, epsilon_v, h):
#     f = 0.5* y**2
#     return f

# def sticking_friction_interpolant_gradient(y, epsilon_v, h):
#     g = y.copy() 
#     return g

# def sticking_friction_interpolant_hessian(y, epsilon_v, h):
    
#     hess = np.ones(y.shape) 
#     return hess


# def stick_slip_friction_interpolant(y, epsilon_v, h):
#     threshold = h * epsilon_v
    
#     below_threshold = (y < threshold).flatten()
    
#     f = y.copy()
#     yl = y[below_threshold]
#     f[below_threshold] = - yl**3/(3 *threshold**2) +  yl**2/threshold + threshold/3
#     return f

# def stick_slip_friction_interpolant_gradient(y, epsilon_v, h):
    
#     threshold = h * epsilon_v
    
#     below_threshold = (y < threshold).flatten()
    
#     yl = y[below_threshold]
#     g = np.ones(y.shape)
#     g[below_threshold] = - yl**2/(threshold**2) + 2 * yl/threshold

#     return g

# def stick_slip_friction_interpolant_hessian(y, epsilon_v, h):
#     threshold = h * epsilon_v
#     below_threshold = (y < threshold).flatten()
#     yl = y[below_threshold]
    
#     h = np.zeros((y.shape[0], 1))
#     h[below_threshold] = - 2 * yl / (threshold**2) + 2 * np.ones(yl.shape)/ threshold
#     # h[below_threshold] = - 2 * yl/threshold**2 + 2/threshold
#     return h
    

# def quadratic_barrier_energy(d, d_hat):
#     d = d.reshape(-1, 1)
#     energy_densities = np.zeros(d.shape)
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
#     e = (dl - d_hat)**2
#     energy_densities[less_than_dhat] = e
#     return energy_densities

# def quadratic_barrier_gradient(d, d_hat):
#     d = d.reshape(-1, 1)
#     grads = np.zeros((d.shape[0], 1))
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
#     g = 2*(dl - d_hat)
#     grads[less_than_dhat] = g

#     return grads

# def quadratic_barrier_hessian(d, d_hat):
#     d = d.reshape(-1, 1)
#     hess = np.zeros((d.shape[0], 1))
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
#     h = 2 * np.ones((dl.shape[0], ))
#     hess[less_than_dhat] = h
#     return hess


# def cubic_barrier_energy(d, d_hat):
#     d = d.reshape(-1, 1)
#     energy_densities = np.zeros(d.shape)
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
#     e = (np.abs(dl - d_hat))**3
#     energy_densities[less_than_dhat] = e
#     return energy_densities

# def cubic_barrier_gradient(d, d_hat):
#     d = d.reshape(-1, 1)
#     grads = np.zeros((d.shape[0], 1))
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
#     g = -3*(np.abs(dl - d_hat))**2
#     grads[less_than_dhat] = g
#     return grads

# def cubic_barrier_hessian(d, d_hat):
#     d = d.reshape(-1, 1)
#     hess = np.zeros((d.shape[0], 1))
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
#     h = 6*(np.abs(dl - d_hat))
#     hess[less_than_dhat] = h
#     return hess


# def ipc_barrier_energy( d, d_hat):
#     d = d.reshape(-1, 1)
#     energy_densities = np.zeros(d.shape)
    
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
#     e = - (dl - d_hat)**2 * np.log(dl / d_hat)
#     energy_densities[less_than_dhat] = e
#     return energy_densities

# def ipc_barrier_gradient(d, d_hat):
#     d = d.reshape(-1, 1)
#     grads = np.zeros((d.shape[0], 1))
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
    
    
#     g = - 2*(dl - d_hat) * np.log(dl / d_hat) + \
#         - (dl - d_hat)**2 *(1.0/dl) 
        
#     grads[less_than_dhat] = g
#     return grads

# def ipc_barrier_hessian(d, d_hat):
#     d = d.reshape(-1, 1)
#     hess = np.zeros((d.shape[0], 1))
#     less_than_dhat = d < d_hat
#     dl = d[less_than_dhat]
    
#     h = - 2 * np.log(dl / d_hat)  \
#         - 4*(dl - d_hat) * (1/dl)  \
#         + (dl - d_hat)**2 *(1/dl)**2
        
#     hess[less_than_dhat] = h
#     return hess
    
    
    
    
"""Barrier and friction density functions for contact.

These are scalar-in / scalar-out building blocks evaluated per contact pair:
each takes a distance ``d`` (and a threshold ``d_hat``) and returns the per-pair
barrier energy density, its gradient, or its Hessian w.r.t. ``d``. They are the
element-level pieces of a contact model; global assembly into ``x`` lives in the
contact code that consumes them.

Three barrier families are provided (``quadratic_*``, ``cubic_*``, ``ipc_*``),
each with the standard energy / gradient / hessian trio, plus friction
interpolants (``sticking_*`` and ``stick_slip_*``).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Friction interpolants                                                       #
# --------------------------------------------------------------------------- #
def sticking_friction_interpolant(y: np.ndarray) -> np.ndarray:
    """Smooth sticking-only friction interpolant ``f(y) = 0.5 * y^2``.

    Parameters
    ----------
    y : np.ndarray (m, 1)
        Relative tangential speed (or its proxy).
    epsilon_v : float
        Velocity threshold parameter.
    h : float
        Timestep.

    Returns
    -------
    f : np.ndarray (m, 1)
        Interpolant value per pair.
    """
    return 0.5 * y ** 2


def sticking_friction_interpolant_gradient(y: np.ndarray) -> np.ndarray:
    """Gradient of :func:`sticking_friction_interpolant` w.r.t. ``y``.

    Parameters
    ----------
    y : np.ndarray (m, 1)
        Relative tangential speed.
    epsilon_v : float
        Velocity threshold parameter.
    h : float
        Timestep.

    Returns
    -------
    g : np.ndarray (m, 1)
        Gradient per pair.
    """
    return y.copy()


def sticking_friction_interpolant_hessian(y: np.ndarray) -> np.ndarray:
    """Hessian of :func:`sticking_friction_interpolant` w.r.t. ``y``.

    Parameters
    ----------
    y : np.ndarray (m, 1)
        Relative tangential speed.
    epsilon_v : float
        Velocity threshold parameter.
    h : float
        Timestep.

    Returns
    -------
    hess : np.ndarray (m, 1)
        Second derivative per pair (constant one).
    """
    return np.ones(y.shape)


def stick_slip_friction_interpolant(y: np.ndarray, epsilon_v: float, h: float) -> np.ndarray:
    """Smooth stick-slip friction interpolant.

    Quadratic-cubic blend below the threshold ``h * epsilon_v`` and linear above.

    Parameters
    ----------
    y : np.ndarray (m, 1)
        Relative tangential speed.
    epsilon_v : float
        Velocity threshold parameter.
    h : float
        Timestep.

    Returns
    -------
    f : np.ndarray (m, 1)
        Interpolant value per pair.
    """
    threshold = h * epsilon_v
    below = (y < threshold).flatten()
    f = y.copy()
    yl = y[below]
    f[below] = -yl ** 3 / (3 * threshold ** 2) + yl ** 2 / threshold + threshold / 3
    return f


def stick_slip_friction_interpolant_gradient(y: np.ndarray, epsilon_v: float, h: float) -> np.ndarray:
    """Gradient of :func:`stick_slip_friction_interpolant` w.r.t. ``y``.

    Parameters
    ----------
    y : np.ndarray (m, 1)
        Relative tangential speed.
    epsilon_v : float
        Velocity threshold parameter.
    h : float
        Timestep.

    Returns
    -------
    g : np.ndarray (m, 1)
        Gradient per pair.
    """
    threshold = h * epsilon_v
    below = (y < threshold).flatten()
    yl = y[below]
    g = np.ones(y.shape)
    g[below] = -yl ** 2 / (threshold ** 2) + 2 * yl / threshold
    return g


def stick_slip_friction_interpolant_hessian(y: np.ndarray, epsilon_v: float, h: float) -> np.ndarray:
    """Hessian of :func:`stick_slip_friction_interpolant` w.r.t. ``y``.

    Parameters
    ----------
    y : np.ndarray (m, 1)
        Relative tangential speed.
    epsilon_v : float
        Velocity threshold parameter.
    h : float
        Timestep.

    Returns
    -------
    hess : np.ndarray (m, 1)
        Second derivative per pair (zero above the threshold).
    """
    threshold = h * epsilon_v
    below = (y < threshold).flatten()
    yl = y[below]
    hess = np.zeros((y.shape[0], 1))
    hess[below] = -2 * yl / (threshold ** 2) + 2 * np.ones(yl.shape) / threshold
    return hess


# --------------------------------------------------------------------------- #
# Quadratic barrier                                                           #
# --------------------------------------------------------------------------- #
def quadratic_barrier_energy(d: np.ndarray, d_hat: float) -> np.ndarray:
    """Quadratic barrier energy density ``(d - d_hat)^2`` for ``d < d_hat``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    energy_densities : np.ndarray (m, 1)
        Per-pair energy density, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    out = np.zeros(d.shape)
    active = d < d_hat
    out[active] = (d[active] - d_hat) ** 2
    return out


def quadratic_barrier_gradient(d: np.ndarray, d_hat: float) -> np.ndarray:
    """Gradient of :func:`quadratic_barrier_energy` w.r.t. ``d``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    grads : np.ndarray (m, 1)
        Per-pair gradient, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    grads = np.zeros((d.shape[0], 1))
    active = d < d_hat
    grads[active] = 2 * (d[active] - d_hat)
    return grads


def quadratic_barrier_hessian(d: np.ndarray, d_hat: float) -> np.ndarray:
    """Hessian of :func:`quadratic_barrier_energy` w.r.t. ``d``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    hess : np.ndarray (m, 1)
        Per-pair second derivative, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    hess = np.zeros((d.shape[0], 1))
    active = d < d_hat
    hess[active] = 2.0
    return hess


# --------------------------------------------------------------------------- #
# Cubic barrier                                                               #
# --------------------------------------------------------------------------- #
def cubic_barrier_energy(d: np.ndarray, d_hat: float) -> np.ndarray:
    """Cubic barrier energy density ``|d - d_hat|^3`` for ``d < d_hat``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    energy_densities : np.ndarray (m, 1)
        Per-pair energy density, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    out = np.zeros(d.shape)
    active = d < d_hat
    out[active] = (np.abs(d[active] - d_hat)) ** 3
    return out


def cubic_barrier_gradient(d: np.ndarray, d_hat: float) -> np.ndarray:
    """Gradient of :func:`cubic_barrier_energy` w.r.t. ``d``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    grads : np.ndarray (m, 1)
        Per-pair gradient, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    grads = np.zeros((d.shape[0], 1))
    active = d < d_hat
    grads[active] = -3 * (np.abs(d[active] - d_hat)) ** 2
    return grads


def cubic_barrier_hessian(d: np.ndarray, d_hat: float) -> np.ndarray:
    """Hessian of :func:`cubic_barrier_energy` w.r.t. ``d``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    hess : np.ndarray (m, 1)
        Per-pair second derivative, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    hess = np.zeros((d.shape[0], 1))
    active = d < d_hat
    hess[active] = 6 * (np.abs(d[active] - d_hat))
    return hess


# --------------------------------------------------------------------------- #
# IPC log barrier                                                             #
# --------------------------------------------------------------------------- #
def ipc_barrier_energy(d: np.ndarray, d_hat: float) -> np.ndarray:
    """IPC log-barrier energy density ``-(d - d_hat)^2 log(d / d_hat)``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    energy_densities : np.ndarray (m, 1)
        Per-pair energy density, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    out = np.zeros(d.shape)
    active = d < d_hat
    dl = d[active]
    out[active] = -(dl - d_hat) ** 2 * np.log(dl / d_hat)
    return out


def ipc_barrier_gradient(d: np.ndarray, d_hat: float) -> np.ndarray:
    """Gradient of :func:`ipc_barrier_energy` w.r.t. ``d``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    grads : np.ndarray (m, 1)
        Per-pair gradient, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    grads = np.zeros((d.shape[0], 1))
    active = d < d_hat
    dl = d[active]
    grads[active] = -2 * (dl - d_hat) * np.log(dl / d_hat) - (dl - d_hat) ** 2 * (1.0 / dl)
    return grads


def ipc_barrier_hessian(d: np.ndarray, d_hat: float) -> np.ndarray:
    """Hessian of :func:`ipc_barrier_energy` w.r.t. ``d``.

    Parameters
    ----------
    d : np.ndarray (m,) or (m, 1)
        Per-pair distances.
    d_hat : float
        Activation distance.

    Returns
    -------
    hess : np.ndarray (m, 1)
        Per-pair second derivative, zero where ``d >= d_hat``.
    """
    d = d.reshape(-1, 1)
    hess = np.zeros((d.shape[0], 1))
    active = d < d_hat
    dl = d[active]
    hess[active] = (
        -2 * np.log(dl / d_hat)
        - 4 * (dl - d_hat) * (1 / dl)
        + (dl - d_hat) ** 2 * (1 / dl) ** 2
    )
    return hess