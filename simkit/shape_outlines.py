"""2D outline generators and Triangle-based meshing from closed polylines."""

from typing import Any, Callable, Tuple

import numpy as np
import scipy.integrate
import igl.triangle

from .closed_polyline import closed_polyline


def shape_mesh(
    outline_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
    flags: str = 'qa0.1',
    *args: Any,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a closed outline into a surface mesh.

    Parameters
    ----------
    outline_func : callable
        Function returning ``(Xs, Es)`` boundary vertices and edges.
    flags : str, optional
        Flags passed to ``igl.triangle.triangulate``.
    *args, **kwargs
        Forwarded to ``outline_func``.

    Returns
    -------
    X : np.ndarray (nv, 2)
        Interior mesh vertices.
    F : np.ndarray (nf, 3)
        Triangle faces.
    Xs : np.ndarray
        Boundary vertices from the outline.
    Es : np.ndarray
        Boundary edges from the outline.
    """
    Xs, Es = outline_func(*args, **kwargs)

    [X, F, _, _, _] = igl.triangle.triangulate(Xs, Es, flags=flags)
    return X, F, Xs, Es


def arrow_outline(
    height_body: float = 1,
    width_body: float = 0.15,
    width_head: float = 0.3,
    height_head: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed polyline outline of an arrow shape.

    Parameters
    ----------
    height_body, width_body : float, optional
        Shaft height and width.
    width_head, height_head : float, optional
        Head width and height above the shaft.

    Returns
    -------
    X : np.ndarray (7, 2)
        Outline vertices.
    E : np.ndarray (7, 2)
        Closed edge list.
    """

    wb_2 = width_body / 2
    wh_2 = width_head / 2
    X = np.array([[-wb_2, 0],
                  [wb_2, 0],
                  [wb_2, height_body],
                  [wh_2, height_body],
                  [0, height_body + height_head],
                  [-wh_2, height_body],
                  [-wb_2, height_body]])

    E = closed_polyline(X)
    return X, E


def circle_outline(radius: float = 1.0, n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Closed polyline approximating a circle.

    Parameters
    ----------
    radius : float, optional
        Circle radius.
    n : int, optional
        Number of vertices on the boundary.

    Returns
    -------
    X : np.ndarray (n, 2)
        Outline vertices.
    E : np.ndarray (n, 2)
        Closed edge list.
    """

    theta = np.linspace(0, 2 * np.pi, n)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    X = np.hstack((x[:, None], y[:, None]))

    E = closed_polyline(X)
    return X, E


def ellipse_outline(a: float = 1, b: float = 0.5, n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Closed polyline with approximately uniform arc-length spacing on an ellipse.

    Parameters
    ----------
    a, b : float, optional
        Semi-axis lengths.
    n : int, optional
        Number of boundary samples.

    Returns
    -------
    X : np.ndarray (n, 2)
        Outline vertices.
    E : np.ndarray (n, 2)
        Closed edge list.
    """

    # Arc length differential for ellipse
    def dS(theta: float) -> float:
        return np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)

    # Total perimeter (approximate by integrating over 0..2pi)
    total_perimeter, _ = scipy.integrate.quad(dS, 0, 2 * np.pi)
    arc_lengths = np.linspace(0, total_perimeter, n, endpoint=False)

    # Precompute cumulative arc length as a function of theta
    thetas = np.linspace(0, 2 * np.pi, 1000)
    cumlen = np.zeros_like(thetas)
    for i in range(1, len(thetas)):
        cumlen[i] = cumlen[i-1] + scipy.integrate.quad(dS, thetas[i-1], thetas[i])[0]

    # For each target arc length, find the corresponding theta
    theta_points = np.interp(arc_lengths, cumlen, thetas)
    x = a * np.cos(theta_points)
    y = b * np.sin(theta_points)
    X = np.hstack((x[:, None], y[:, None]))
    E = closed_polyline(X)
    return X, E


def rectangle_outline(small: float = 0.5, large: float = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned rectangle outline.

    Parameters
    ----------
    small, large : float, optional
        Half-heights along the short and long sides.

    Returns
    -------
    X : np.ndarray (4, 2)
        Corner vertices.
    E : np.ndarray (4, 2)
        Closed edge list.
    """

    # make a rectangle
    X = np.array([[-large, -small], [large, -small], [large, small], [-large, small]])
    E = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    return X, E


def plus_sign_outline(small: float = 0.5, large: float = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Plus-sign outline (12 vertices).

    Parameters
    ----------
    small, large : float, optional
        Inner and outer extent of the arms (requires ``small < large``).

    Returns
    -------
    X : np.ndarray (12, 2)
        Outline vertices.
    E : np.ndarray (12, 2)
        Closed edge list.
    """
    assert small < large
    # make a plus sign
    X = np.array([[small, small], [large, small], [large, -small], [small, -small],
                [small, -large], [-small, -large], [-small, -small],
                [-large, -small], [-large, small], [-small, small],
                [-small, large], [small, large]])

    E = np.hstack((np.arange(12)[:, None], np.roll(np.arange(12)[:, None], -1, axis=0)))

    return X, E


def star_sign_outline(
    small: float = 0.5,
    large: float = 5,
    legs: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Star outline with alternating inner and outer vertices.

    Parameters
    ----------
    small, large : float, optional
        Inner and outer radii of the star tips.
    legs : int, optional
        Number of star points.

    Returns
    -------
    X : np.ndarray (3*legs, 2)
        Outline vertices.
    E : np.ndarray (3*legs, 2)
        Closed edge list.
    """

    angle = 2 * np.pi / legs

    X = []

    for i in range(legs):

        theta = angle * i - angle/2
        theta_next = angle * (i + 1) - angle/2
        pos = np.array([np.cos(theta), np.sin(theta)]) * small
        pos_next = np.array([np.cos(theta_next), np.sin(theta_next)]) * small
        disp = pos_next - pos
        orthog = np.array([-disp[1], disp[0]])
        orthog /= -np.linalg.norm(orthog)

        X.append(pos)
        X.append(pos + orthog * large)
        X.append(pos + orthog * large + disp)

    X = np.array(X)
    E = np.hstack((np.arange(3 * legs)[:, None], np.roll(np.arange(3 * legs)[:, None], -1, axis=0)))

    return X, E
