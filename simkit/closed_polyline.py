import numpy as np

def closed_polyline(V):
    """
    Create a closed polyline from a set of vertices assuming the last vertex is connected to the first vertex
    .
    
    Parameters
    ----------
    V (n, d) array of vertex positions
    
    Returns
    -------
    E (m, 2) array of edge indices
    
    Example
    -------
    ```python
    V = np.random.rand(100, 3)
    E = closed_polyline(V)
    ```
    
    In the above, E will have the form:
    E = [[0 1], [1 2], [2 3], ..., [99 0]]
    
    This means that the polyline is closed and the last vertex is connected to the first vertex.
    """
    
    E = np.hstack((np.arange(V.shape[0])[:, None], np.arange(V.shape[0])[:, None] + 1))
    E[-1, -1] = 0
    return E
