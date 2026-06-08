"""Concatenate several meshes into a single vertex/connectivity pair."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def combine_meshes(
    V_list: Sequence[np.ndarray], F_list: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a list of meshes into one mesh by stacking and re-indexing.

    Each input mesh ``i`` is a pair ``(V_list[i], F_list[i])`` where the
    connectivity ``F_list[i]`` indexes into its own vertex block ``V_list[i]``.
    The vertices are stacked into one array and every connectivity index is
    shifted by the number of vertices preceding its mesh, so the returned
    connectivity indexes consistently into the stacked vertex array. No
    deduplication of coincident vertices is performed.

    Works in any dimension: ``V`` may have 2 (planar) or 3 (spatial) columns,
    and the connectivity may be edges (2 columns), triangles (3 columns), etc.
    All meshes must share the same vertex dimension and the same number of
    connectivity columns.

    Parameters
    ----------
    V_list : sequence of np.ndarray
        Per-mesh vertex arrays, each ``(n_i, d)`` with a common ``d``.
    F_list : sequence of np.ndarray
        Per-mesh connectivity arrays, each ``(m_i, s)`` with a common ``s``,
        indexing into the matching ``V_list[i]``.

    Returns
    -------
    V : np.ndarray (sum(n_i), d)
        Vertices of every mesh stacked in input order.
    F : np.ndarray (sum(m_i), s)
        Connectivity of every mesh stacked in input order, with indices
        offset to address the stacked vertex array.

    Raises
    ------
    ValueError
        If ``V_list`` and ``F_list`` differ in length, or are empty.

    Example
    -------
    ```python
    V0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    V1 = np.array([[0.0, 1.0], [1.0, 1.0]])
    E0 = np.array([[0, 1]])
    E1 = np.array([[0, 1]])
    V, E = combine_meshes([V0, V1], [E0, E1])
    # V has 4 rows; E == [[0, 1], [2, 3]]  (second edge shifted by 2)
    ```
    """
    if len(V_list) != len(F_list):
        raise ValueError(
            f"combine_meshes got {len(V_list)} vertex arrays but "
            f"{len(F_list)} connectivity arrays"
        )
    if len(V_list) == 0:
        raise ValueError("combine_meshes needs at least one mesh")

    Vs = [np.asarray(V) for V in V_list]
    Fs = [np.asarray(F) for F in F_list]

    # Per-mesh vertex counts -> the index offset applied to each mesh's
    # connectivity is the running total of vertices in the meshes before it.
    vcounts = np.fromiter((V.shape[0] for V in Vs), dtype=np.int64, count=len(Vs))
    offsets = np.concatenate(([0], np.cumsum(vcounts)[:-1]))

    # Stack everything, then add the offsets in one vectorized shot: repeat
    # each mesh's offset across its own rows so it lines up with the stacked
    # connectivity, and broadcast across the index columns.
    fcounts = np.fromiter((F.shape[0] for F in Fs), dtype=np.int64, count=len(Fs))
    V = np.concatenate(Vs, axis=0)
    F = np.concatenate(Fs, axis=0) + np.repeat(offsets, fcounts)[:, None]
    return V, F
