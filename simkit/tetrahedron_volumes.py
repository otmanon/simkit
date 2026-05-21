import numpy as np

def tetrahedron_volumes(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    e = X[T[:, 1:]] - X[T[:, [0]]]   # (m, 3, 3): rows e1, e2, e3
    return (np.linalg.det(e)) / 6.0