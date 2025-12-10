import numpy as np
import scipy as sp


def hinge_jacobian_compact(X, H):
    Ax, Ay = X[H[:,0], 0], X[H[:,0], 1]
    Bx, By = X[H[:,1], 0], X[H[:,1], 1]
    Cx, Cy = X[H[:,2], 0], X[H[:,2], 1]
    
        
    # Numerical regularization parameter
    epsilon = 1e-6
    
    # Precompute common denominators with regularization
    dist_AB_sq = np.maximum(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2, epsilon)
    dist_BC_sq = np.maximum(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2, epsilon)
    
    if np.any(dist_AB_sq == 0):
        print("WARNING:dist_AB_sq is 0, leading to a divide by zero in the hinge jacobian")
    if np.any(dist_BC_sq == 0):
        print("WARNING: dist_BC_sq is 0, leading to a divide by zero in the hinge jacobian")



    J = np.array([(Ay - By)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2), 
            (-Ax + Bx)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2),
            (-(Ay - Cy)*((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy)) - ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*(Ax - 2*Bx + Cx))/(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2), 
            ((Ax - Cx)*((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy)) - ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*(Ay - 2*By + Cy))/(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2), 
            (By - Cy)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2), 
            (-Bx + Cx)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)]).T
    return J


def hinge_jacobian(X, H):
    J_compact = hinge_jacobian_compact(X, H)
    
    rows = np.repeat(np.arange(H.shape[0])[:, None], 6, axis=1)
    cols = np.repeat(H*2, 2, axis=1) + np.array([[0, 1, 0, 1, 0, 1]])
    
    J = sp.sparse.csc_matrix((J_compact.flatten(), (rows.flatten(), cols.flatten())),
                             shape=(H.shape[0], 2*X.shape[0]))
    return J
