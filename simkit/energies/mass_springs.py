
import numpy as np
import scipy as sp

def mass_springs_energy_x(x, E, mu, vol, l0):
    """
    Mass-springs energy given vertex positions.

    Parameters
    ----------
    x : (num_vertices, dim) np.ndarray
        Vertex positions.
    E : (num_edges, 2) np.ndarray
        Undirected edges as pairs of vertex indices (i, j).
    mu : (num_edges, 1) np.ndarray
        Spring stiffness parameter per edge.
    vol : (num_edges, 1) np.ndarray
        Volume/weight per edge.
    l0 : (num_edges, 1) np.ndarray
        Rest length per edge.
    """
    d = x[E[:, 1]] - x[E[:, 0]]
    return mass_springs_energy_d(d, mu, vol, l0)

def mass_springs_energy_z(z, J, mu, vol, l0, Jx0=None):
    if Jx0 is not None:
        d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
    else:
        d = (J @ z).reshape(l0.shape[0], -1)
    e = mass_springs_energy_d(d, mu, vol, l0)
    return e

def mass_springs_gradient_dz(z, J, mu, vol, l0, Jx0=None):
    if Jx0 is not None:
        d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
    else:
        d = (J @ z).reshape(l0.shape[0], -1)
    dedd= mass_springs_gradient_dd(d, mu, vol, l0)
    g = J.T @ (dedd.reshape(-1, 1))
    return g

def mass_springs_hessian_d2z(z, J, mu, vol, l0, Jx0=None):
    if Jx0 is not None:
        d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
    else:
        d = (J @ z).reshape(l0.shape[0], -1)
    de2dd2 = mass_springs_hessian_dd(d, mu, vol, l0)
    Q = sp.sparse.block_diag(de2dd2)
    H = J.T @ Q @ J
    return H



def mass_springs_energy_d(d, ym, vol, l0):
    """
    Parameters
    ----------
    d : (num_edges, dim) np.ndarray
        per edge displacements
    ym : (num_edges, 1) np.ndarray
        The mu parameter of the system.
    vol : (num_edges, 1) np.ndarray
        The volume of the springs.
    l0 : (num_edges, 1) np.ndarray
        The rest length of the springs.
    """
    l0 = np.asarray(l0).reshape(-1, 1)
    ym = np.asarray(ym).reshape(-1, 1)
    vol = np.asarray(vol).reshape(-1, 1)
    
    l = np.linalg.norm(d, axis=1)[:, None]
    coeff = vol * ym / (l0**2)
    e = 0.5 * np.sum(coeff * (l - l0)**2)
    e = e.reshape(-1, 1)
    return e


def mass_springs_gradient_dd(d, ym, vol, l0):
    l0 = np.asarray(l0).reshape(-1, 1)
    ym = np.asarray(ym).reshape(-1, 1)
    vol = np.asarray(vol).reshape(-1, 1)

    l = np.linalg.norm(d, axis=1)[:, None]
    
    coeff = vol * ym / (l0**2)
    g = coeff * (l - l0) * d / l
    return g

def mass_springs_hessian_dddl0(d, ym, vol, l0):
    l0 = np.asarray(l0).reshape(-1, 1)
    ym = np.asarray(ym).reshape(-1, 1)
    vol = np.asarray(vol).reshape(-1, 1)

    l = np.linalg.norm(d, axis=1)[:, None]
   
    coeff = vol * ym * d / l 
    # g = coeff * (l/ (l0**2) - (1/l0))
    
    dg_dl0 = coeff * (1/l0**2 - 2*l/ (l0**3) )
    
    
    # import simkit as sk
    # energy_func = lambda l: mass_springs_gradient_dd(d, ym, vol, l).reshape(-1, 1)
    # g_fd =sk.gradient_cfd(energy_func, l0, 1e-6)[:, 0, :]
    
    # Q = sp.sparse.block_diag(dg_dl0[:, None, :])
    
    # error = np.linalg.norm(Q - g_fd.T)
    # print(error)
    return dg_dl0

def mass_springs_hessian_dd(d, ym, vol, l0):
    l0 = np.asarray(l0).reshape(-1, 1)
    ym = np.asarray(ym).reshape(-1, 1)
    vol = np.asarray(vol).reshape(-1, 1)
    l = np.linalg.norm(d, axis=1)[:, None]
    l3 = l**3
    ddT = d[ :, :, None] @ d[:, None, :]
    I = np.eye(d.shape[1])[None, :, :]
    term = I - l0[:, None, :] * (I / l[ :, None, :] - ddT / l3[ :, None, :])
    
    coeff = vol * ym / (l0**2)
    hess = coeff[:, None, :] * (term) 
    H = hess 
    return H


def mass_springs_energy_l(length, ym, vol, length0):
    """
    Parameters
    ----------
    z : np.ndarray
        The current state of the system.
    mu : np.ndarray
        The mu parameter of the system.
    vol : np.ndarray
        The volume of the springs.
    length : np.ndarray
        The rest length of the springs.
    """
    coeff = vol * ym / (length0**2)
    e = 0.5 * np.sum(coeff * (length - length0)**2)
    e = e.reshape(-1, 1)
    return e


def mass_springs_gradient_dl(length, ym, vol, length0):
    dim = length.shape[1]
    coeff = vol * ym / (length0**2)
    g = coeff * (length - length0) 
    g = g.reshape(-1, dim)    
    return g


def mass_springs_hessian_d2l(ym, vol, length0):
    coeff = vol * ym / (length0**2)
    H = sp.sparse.diags(np.array(coeff).flatten(), 0)    
    return H



