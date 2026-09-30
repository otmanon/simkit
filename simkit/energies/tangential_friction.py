"""Lagged viscous friction ``E = 1/2 sum_j w_j |(I - n_j n_j^T)(p_j - p0_j)|^2``, ``p = (S x).reshape(-1, dim)``:
sliding away from per-step anchors ``p0`` along the contact surface costs, motion along the normal
``n`` is free. With anchors, normals and active rows of ``S`` frozen per step it is exactly quadratic."""
import numpy as np
import scipy as sp


def _weighted_projectors(n, w):
    T = np.eye(n.shape[1])[None] - n[:, :, None] * n[:, None, :]
    return sp.sparse.block_diag(list(w[:, None, None] * T), format="csr")


def tangential_friction_energy_x(x, S, p0, n, w):
    r = S @ np.ravel(x) - np.ravel(p0)
    return float(0.5 * r @ (_weighted_projectors(n, w) @ r))


def tangential_friction_gradient_x(x, S, p0, n, w):
    return S.T @ (_weighted_projectors(n, w) @ (S @ np.ravel(x) - np.ravel(p0))).reshape(-1, 1)


def tangential_friction_hessian_x(x, S, p0, n, w):
    return (S.T @ _weighted_projectors(n, w) @ S).tocsr()
