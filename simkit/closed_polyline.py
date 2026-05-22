"""Build closed-loop edge connectivity from an ordered list of vertices."""

import numpy as np


def closed_polyline(V: np.ndarray) -> np.ndarray:
    """Edge list for a closed polyline through consecutive vertices.

    Connects vertex ``i`` to ``i + 1`` for every vertex, then wraps the last
    vertex back to the first so the loop is closed.

    Parameters
    ----------
    V : np.ndarray (n, d)
        Vertex positions. Only the count ``n`` is used; coordinates are ignored.

    Returns
    -------
    E : np.ndarray (n, 2)
        Edge index pairs ``[[0, 1], [1, 2], ..., [n-1, 0]]``.

    Example
    -------
    ```python
    V = np.random.rand(100, 3)
    E = closed_polyline(V)   # E[-1] == [99, 0]
    ```
    """
    n = V.shape[0]
    starts = np.arange(n)[:, None]       # [0, 1, ..., n-1] as a column
    E = np.hstack((starts, starts + 1))  # each edge goes i -> i+1
    E[-1, -1] = 0                        # wrap the final edge back to vertex 0
    return E