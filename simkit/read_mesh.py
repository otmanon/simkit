"""Reader for MEDIT ``.mesh`` tetrahedral / triangle meshes (pure NumPy)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def read_mesh(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a MEDIT ``.mesh`` file.

    A NumPy-only replacement for ``igl.read_mesh`` covering the ASCII subset
    the ``data/`` submodule uses: ``Vertices``, ``Triangles`` and
    ``Tetrahedra`` sections with 1-based indices and a trailing reference
    tag per entry.

    Parameters
    ----------
    path : str
        Path to the ``.mesh`` file.

    Returns
    -------
    X : np.ndarray (n, dim)
        Vertex positions (``dim`` from the file's ``Dimension`` line).
    T : np.ndarray (t, 4)
        Tetrahedra, 0-based. Empty ``(0, 4)`` if the file has none.
    F : np.ndarray (f, 3)
        Triangles, 0-based. Empty ``(0, 3)`` if the file has none.
    """
    with open(path, "r") as fh:
        tokens = fh.read().split()

    def section(name: str, width: int) -> np.ndarray:
        if name not in tokens:
            return np.zeros((0, width - 1), dtype=np.int64)
        i = tokens.index(name)
        count = int(tokens[i + 1])
        block = np.array(tokens[i + 2 : i + 2 + width * count], dtype=np.float64).reshape(count, width)
        return block

    dim = 3
    if "Dimension" in tokens:
        i = tokens.index("Dimension")
        dim = int(tokens[i + 1])

    verts = section("Vertices", dim + 1)
    X = np.asarray(verts[:, :dim], dtype=np.float64)
    T = section("Tetrahedra", 5)[:, :4].astype(np.int64) - 1
    F = section("Triangles", 4)[:, :3].astype(np.int64) - 1
    return X, T, F
