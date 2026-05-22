"""Per-triangle area-weighted normals and their derivatives.

The "area normal" of a triangle is the cross product of two edges; its
magnitude is twice the triangle area and its direction is the face normal.
These three functions provide the area normal, its gradient with respect to
the nine corner coordinates, and its (constant) Hessian.
"""

import numpy as np


def area_normal_element(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Per-triangle area normal ``n = e1 x (-e2)``.

    Parameters
    ----------
    x0, x1, x2 : np.ndarray (m, 3)
        The three corner positions of each triangle.

    Returns
    -------
    n : np.ndarray (m, 3)
        Area normal per triangle; ``|n|`` is twice the triangle area.
    """
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x1 - x0

    n = np.cross(e1, -e2)
    return n


def area_normal_gradient_element(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Gradient of the area normal w.r.t. the nine corner coordinates.

    Because the area normal is a cross product of edge differences, its
    derivative w.r.t. each corner is a skew-symmetric (cross-product) matrix of
    the opposite edge.

    Parameters
    ----------
    x0, x1, x2 : np.ndarray (m, 3)
        The three corner positions of each triangle.

    Returns
    -------
    dn_dx : np.ndarray (m, 3, 9)
        Derivative of the 3-vector normal w.r.t. the stacked corners
        ``[x0, x1, x2]``.
    """

    def skew(x: np.ndarray) -> np.ndarray:
        """Stack of skew-symmetric (cross-product) matrices for each row of x.

        For a vector ``v``, ``skew(v) @ w == cross(v, w)``.
        """
        z = np.zeros((x.shape[0]))
        skew_matrix = np.array([[z, -x[:, 2], x[:, 1]],
                                [x[:, 2], z, -x[:, 0]],
                                [-x[:, 1], x[:, 0], z]])
        return skew_matrix.transpose(2, 1, 0)

    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x1 - x0

    # Each corner's contribution is the skew matrix of the opposite edge.
    dn_dx0 = skew(e0)
    dn_dx1 = skew(e1)
    dn_dx2 = skew(e2)

    dn_dx = np.concatenate([dn_dx0, dn_dx1, dn_dx2], axis=2)
    return dn_dx


def area_normal_hessian_element(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Hessian of the area normal w.r.t. the nine corner coordinates.

    The area normal is bilinear in the corner positions, so its Hessian is a
    constant array of 0, +/-1 entries that does not depend on the geometry. It
    is tiled once per triangle. (The constant block was originally verified by
    finite differences.)

    Parameters
    ----------
    x0, x1, x2 : np.ndarray (m, 3)
        The three corner positions of each triangle. Only ``x0.shape[0]`` (the
        triangle count) is used; the values themselves do not matter.

    Returns
    -------
    H : np.ndarray (m, 3, 9, 9)
        Per-triangle Hessian of each normal component over the nine DOFs.
    """
    # Constant second-derivative block, one 9x9 slab per normal component.
    H = np.array([[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.],
                    [ 0.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., -1.],
                    [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.],
                    [ 0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.]],   # d2 n_x

                   [[ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., -1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.],
                    [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 1.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [-1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]],   # d2 n_y

                   [[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
                    [-1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                    [ 1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]])  # d2 n_z

    num_elem = x0.shape[0]
    H = np.tile(H, (num_elem, 1, 1, 1))      # broadcast the constant per triangle
    return H