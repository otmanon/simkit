import numpy as np
import igl.triangle
from .closed_polyline import closed_polyline



def shape_mesh(outline_func, flags='qa0.1', *args, **kwargs):
    """
    Create a mesh from an outline function.
    
    Args:
        outline_func (function): Function to create the outline of the mesh.
        *args: Positional arguments for the outline function.
        **kwargs: Keyword arguments for the outline function.
    
    Returns:
        tuple: Tuple containing vertices and edges of the mesh.
    """
    Xs, Es = outline_func(*args, **kwargs)

    [X, F, _, _, _] = igl.triangle.triangulate(Xs, Es, flags=flags)
    return X, F, Xs, Es

def arrow_outline(height_body=1, width_body=0.15, width_head=0.3, height_head=0.3):

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


def circle_outline(radius=1.0, n=100):

    theta = np.linspace(0, 2 * np.pi, n)


    x = radius * np.cos(theta) 
    y = radius * np.sin(theta) 

    X = np.hstack((x[:, None], y[:, None]))

    E = closed_polyline(X)
    return X, E

import scipy.integrate
import scipy.optimize
def ellipse_outline(a=1, b=0.5, n=100):
    """
    Returns n points regularly spaced around the perimeter of an ellipse with axes a and b.
    """

    # Arc length differential for ellipse
    def dS(theta):
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

def rectangle_outline(small=0.5, large=2):
  
    # make a rectangle
    X = np.array([[-large, -small], [large, -small], [large, small], [-large, small]])    
    E = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    return X, E
def plus_sign_outline(small=0.5, large=2):
    assert small < large
    # make a plus sign
    X = np.array([[small, small], [large, small], [large, -small], [small, -small], 
                [small, -large], [-small, -large], [-small, -small], 
                [-large, -small], [-large, small], [-small, small], 
                [-small, large], [small, large]])
    
    E = np.hstack((np.arange(12)[:, None], np.roll(np.arange(12)[:, None], -1, axis=0)))

    return X, E


def star_sign_outline(small=0.5, large=5, legs=10):

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
        



    