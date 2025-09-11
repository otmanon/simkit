import numpy as np


def gradient_cfd(phi, y, h):
    """
    Computes the finite difference gradient of a function f, with respect to each of the parameters y
    if phi spits out a tensor of shape (dim_1, dim_2, ..., dim_d), then gradient_cfd will spit out a tensor of shape (dim_1, dim_2, ..., dim_n, dim(y))
    """
    y0 = y.copy()

    phi0 = phi(y)
    g = np.zeros(phi0.shape + (y.shape[0],))
    for i in range(y.shape[0]):
        yib = y0.copy()
        yif = y0.copy()
        yib[i] -= h
        yif[i] += h
        phi_b = phi(yib)
        phi_f = phi(yif)

        g[..., i] = (phi_f - phi_b) / (2 * h)  # the dots mean only inde

    return g