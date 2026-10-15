"""Assembly of per-element stresses and Hessian blocks into global quantities.

The element tier of every energy produces a per-element stress ``P`` (the
gradient w.r.t. the flattened deformation gradient) and per-element Hessian
blocks ``He``. These four flat functions apply the operator that maps the
primary variable to the element quantities, in the full space (a sparse
``J``) or in a subspace (a dense reduced operator ``JB = G J B``)::

    g = J.T  @ vec(P)            assembled_gradient(J, P)
    Q = J.T  @ blockdiag(He) @ J assembled_hessian(J, He)
    g = JB.T @ vec(P)            assembled_subspace_gradient(JB, P)
    Q = JB.T @ blockdiag(He) @ JB assembled_subspace_hessian(JB, He)

``P`` is ``(t, dim, dim)`` or ``(t, b)``, ``He`` is ``(t, b, b)``; both are
expected already weighted by the quadrature weights (and ``He`` PSD-projected
if desired), exactly as the ``*_hessian_x`` tiers do before assembling. On
the C++ backend the sparse assembly produces canonical CSR directly and the
dense ones are chunked over elements across the worker threads.
"""

from __future__ import annotations

import numpy as np
import scipy as sp

from ._backend import _cpp, use_cpp
from .block_diag_sandwich import block_diag_sandwich


def assembled_gradient(J: sp.sparse.spmatrix, P: np.ndarray) -> np.ndarray:
    """``J.T @ vec(P)``: gradient w.r.t. the full-space variable, ``(m, 1)``."""
    P = np.asarray(P, dtype=np.float64)
    if J.shape[0] != P.size:
        raise ValueError("assembled_gradient: J has %d rows but P has %d entries" % (J.shape[0], P.size))
    if use_cpp() and sp.sparse.issparse(J):
        return np.asarray(_cpp.assembled_gradient(J, P.reshape(P.shape[0], -1)))
    return np.asarray(J.T @ P.reshape(-1, 1))


def assembled_hessian(J: sp.sparse.spmatrix, He: np.ndarray) -> sp.sparse.spmatrix:
    """``J.T @ block_diag(He) @ J``: Hessian w.r.t. the full-space variable, sparse ``(m, m)``."""
    He = np.asarray(He, dtype=np.float64)
    if use_cpp() and sp.sparse.issparse(J):
        if J.shape[0] != He.shape[0] * He.shape[1]:
            raise ValueError("assembled_hessian: J has %d rows but He has %d blocks of size %d" % (J.shape[0], He.shape[0], He.shape[1]))
        return _cpp.assembled_hessian(J, He)
    return block_diag_sandwich(J, He)


def assembled_subspace_gradient(JB: np.ndarray, P: np.ndarray) -> np.ndarray:
    """``JB.T @ vec(P)``: reduced gradient, ``(r, 1)`` for a dense ``JB`` of shape ``(t*b, r)``."""
    P = np.asarray(P, dtype=np.float64)
    if JB.shape[0] != P.size:
        raise ValueError("assembled_subspace_gradient: JB has %d rows but P has %d entries" % (JB.shape[0], P.size))
    if use_cpp() and not sp.sparse.issparse(JB):
        return np.asarray(_cpp.assembled_subspace_gradient(np.ascontiguousarray(JB, dtype=np.float64), P.reshape(P.shape[0], -1)))
    return np.asarray(JB.T @ P.reshape(-1, 1))


def assembled_subspace_hessian(JB: np.ndarray, He: np.ndarray) -> np.ndarray:
    """``JB.T @ block_diag(He) @ JB``: reduced Hessian, dense ``(r, r)``."""
    He = np.asarray(He, dtype=np.float64)
    if use_cpp() and not sp.sparse.issparse(JB):
        if JB.shape[0] != He.shape[0] * He.shape[1]:
            raise ValueError("assembled_subspace_hessian: JB has %d rows but He has %d blocks of size %d" % (JB.shape[0], He.shape[0], He.shape[1]))
        return np.asarray(_cpp.assembled_subspace_hessian(np.ascontiguousarray(JB, dtype=np.float64), He))
    return np.asarray(block_diag_sandwich(JB, He))
