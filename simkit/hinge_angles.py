
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