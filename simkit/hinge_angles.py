
import numpy as np

def hinge_angles(X, H):
    # Get vertex positions for all hinges
    A = X[H[:,0]]  # (m,2)
    B = X[H[:,1]]  # (m,2) 
    C = X[H[:,2]]  # (m,2)

    # Compute vectors
    v1 = B - A  # (m,2)
    v2 = C - B  # (m,2)

    # Compute angle between vectors using cross product and dot product
    c = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1]  # dot product
    s = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]  # cross product (z component)
    theta = np.arctan2(s, c)
    return theta.reshape(-1,1)


def hinge_angle_velocities(X, V, H):
    # Positions
    A = X[H[:,0]]
    B = X[H[:,1]]
    C = X[H[:,2]]

    # Velocities
    VA = V[H[:,0]]
    VB = V[H[:,1]]
    VC = V[H[:,2]]

    # Edges from hinge (at B)
    e1 = A - B          # B -> A
    e2 = C - B          # B -> C

    # Time-derivatives of edges
    de1 = VA - VB
    de2 = VC - VB

    # Squared lengths
    L1 = np.sum(e1*e1, axis=1)
    L2 = np.sum(e2*e2, axis=1)

    # Angular velocities of each edge about B:
    # omega = e_perp · de / ||e||^2, with e_perp = (-ey, ex)
    w1 = (-e1[:,1] * de1[:,0] + e1[:,0] * de1[:,1]) / L1
    w2 = (-e2[:,1] * de2[:,0] + e2[:,0] * de2[:,1]) / L2

    theta_dot = w2 - w1
    return theta_dot.reshape(-1, 1)