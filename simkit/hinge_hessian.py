import numpy as np
import scipy as sp
def hinge_hessian_compact(X, H):
    Ax, Ay = X[H[:,0], 0], X[H[:,0], 1]
    Bx, By = X[H[:,1], 0], X[H[:,1], 1]
    Cx, Cy = X[H[:,2], 0], X[H[:,2], 1]
    
    # Numerical regularization parameter
    epsilon = 0
    

    # Precompute common denominators with regularization
    dist_AB_sq = np.maximum(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2, epsilon)
    dist_BC_sq = np.maximum(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2, epsilon)
    
    dot_prod_AB_BC = (Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy)
    cross_prod_AB_BC = (Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx)
    complex_denom = np.maximum(dot_prod_AB_BC**2 + cross_prod_AB_BC**2, epsilon)
    
    if np.any(dist_AB_sq == 0):
        print("WARNING: dist_AB_sq is 0, leading to a divide by zero in the hinge hessian")
    if np.any(dist_BC_sq == 0):
        print("WARNING: dist_BC_sq is 0, leading to a divide by zero in the hinge hessian")
    if np.any(complex_denom == 0):
        print("WARNING: complex_denom is 0, leading to a divide by zero in the hinge hessian")
        
    # Fourth-order denominators
    denom_AB_4th = np.maximum(dist_AB_sq**2, epsilon)
    denom_BC_4th = np.maximum(dist_BC_sq**2, epsilon)
    
    Z = np.zeros((Ax.shape[0]))
    H = np.array([
        [-2*(Ax - Bx)*(Ay - By)/dist_AB_sq**2,
         (-Ax**2 + 2*Ax*Bx - Ay**2 + 2*Ay*By - Bx**2 - By**2 + 2*(Ax - Bx)**2)/dist_AB_sq**2,
         2*(Ax*Ay - Ax*By - Ay*Bx + Bx*By)/denom_AB_4th,
         (-Ax**2 + 2*Ax*Bx + Ay**2 - 2*Ay*By - Bx**2 + By**2)/denom_AB_4th,
         Z,
         Z],
        [(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2 - 2*(Ay - By)**2)/dist_AB_sq**2,
         2*(Ax - Bx)*(Ay - By)/dist_AB_sq**2,
         (-Ax**2 + 2*Ax*Bx + Ay**2 - 2*Ay*By - Bx**2 + By**2)/denom_AB_4th,
         2*(-Ax*Ay + Ax*By + Ay*Bx - Bx*By)/denom_AB_4th,
         Z,
         Z],
        [2*(Ax - Bx)*(Ay - By)/dist_AB_sq**2,
         (Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2 - 2*(Ax - Bx)**2)/dist_AB_sq**2,
         2*(((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*complex_denom - ((Ay - Cy)*dot_prod_AB_BC + cross_prod_AB_BC*(Ax - 2*Bx + Cx))*((Ay - Cy)*cross_prod_AB_BC - dot_prod_AB_BC*(Ax - 2*Bx + Cx)))/complex_denom**2,
         (2*((Ax - Cx)*dot_prod_AB_BC - cross_prod_AB_BC*(Ay - 2*By + Cy))*((Ay - Cy)*cross_prod_AB_BC - dot_prod_AB_BC*(Ax - 2*Bx + Cx)) + ((Ax - Cx)*(Ax - 2*Bx + Cx) + (Ay - Cy)*(Ay - 2*By + Cy))*complex_denom)/complex_denom**2,
         -2*(Bx - Cx)*(By - Cy)/dist_BC_sq**2,
         (-Bx**2 + 2*Bx*Cx - By**2 + 2*By*Cy - Cx**2 - Cy**2 + 2*(Bx - Cx)**2)/dist_BC_sq**2],
        [(-Ax**2 + 2*Ax*Bx - Ay**2 + 2*Ay*By - Bx**2 - By**2 + 2*(Ay - By)**2)/dist_AB_sq**2,
         -2*(Ax - Bx)*(Ay - By)/dist_AB_sq**2,
         (2*((Ax - Cx)*cross_prod_AB_BC + dot_prod_AB_BC*(Ay - 2*By + Cy))*((Ay - Cy)*dot_prod_AB_BC + cross_prod_AB_BC*(Ax - 2*Bx + Cx)) - ((Ax - Cx)*(Ax - 2*Bx + Cx) + (Ay - Cy)*(Ay - 2*By + Cy))*complex_denom)/complex_denom**2,
         2*(cross_prod_AB_BC*complex_denom - ((Ax - Cx)*dot_prod_AB_BC - cross_prod_AB_BC*(Ay - 2*By + Cy))*((Ax - Cx)*cross_prod_AB_BC + dot_prod_AB_BC*(Ay - 2*By + Cy)))/complex_denom**2,
         (Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2 - 2*(By - Cy)**2)/dist_BC_sq**2,
         2*(Bx - Cx)*(By - Cy)/dist_BC_sq**2],
        [Z,
         Z,
         2*(-Bx*By + Bx*Cy + By*Cx - Cx*Cy)/denom_BC_4th,
         (Bx**2 - 2*Bx*Cx - By**2 + 2*By*Cy + Cx**2 - Cy**2)/denom_BC_4th,
         2*(Bx - Cx)*(By - Cy)/dist_BC_sq**2,
         (Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2 - 2*(Bx - Cx)**2)/dist_BC_sq**2],
        [Z,
         Z,
         (Bx**2 - 2*Bx*Cx - By**2 + 2*By*Cy + Cx**2 - Cy**2)/denom_BC_4th,
         2*(Bx*By - Bx*Cy - By*Cx + Cx*Cy)/denom_BC_4th,
         (-Bx**2 + 2*Bx*Cx - By**2 + 2*By*Cy - Cx**2 - Cy**2 + 2*(By - Cy)**2)/dist_BC_sq**2,
         -2*(Bx - Cx)*(By - Cy)/dist_BC_sq**2]
    ])

    return H.transpose(2, 0, 1)



# def hinge_hessian_compact_nonstable(X, H):
#     Ax, Ay = X[H[:,0], 0], X[H[:,0], 1]
#     Bx, By = X[H[:,1], 0], X[H[:,1], 1]
#     Cx, Cy = X[H[:,2], 0], X[H[:,2], 1]
    
#     Z = np.zeros((Ax.shape[0]))
  

#     H = np.array([
#         [-2*(Ax - Bx)*(Ay - By)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2)**2,
#          (-Ax**2 + 2*Ax*Bx - Ay**2 + 2*Ay*By - Bx**2 - By**2 + 2*(Ax - Bx)**2)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2)**2,
#          2*(Ax*Ay - Ax*By - Ay*Bx + Bx*By)/(Ax**4 - 4*Ax**3*Bx + 2*Ax**2*Ay**2 - 4*Ax**2*Ay*By + 6*Ax**2*Bx**2 + 2*Ax**2*By**2 - 4*Ax*Ay**2*Bx + 8*Ax*Ay*Bx*By - 4*Ax*Bx**3 - 4*Ax*Bx*By**2 + Ay**4 - 4*Ay**3*By + 2*Ay**2*Bx**2 + 6*Ay**2*By**2 - 4*Ay*Bx**2*By - 4*Ay*By**3 + Bx**4 + 2*Bx**2*By**2 + By**4),
#          (-Ax**2 + 2*Ax*Bx + Ay**2 - 2*Ay*By - Bx**2 + By**2)/(Ax**4 - 4*Ax**3*Bx + 2*Ax**2*Ay**2 - 4*Ax**2*Ay*By + 6*Ax**2*Bx**2 + 2*Ax**2*By**2 - 4*Ax*Ay**2*Bx + 8*Ax*Ay*Bx*By - 4*Ax*Bx**3 - 4*Ax*Bx*By**2 + Ay**4 - 4*Ay**3*By + 2*Ay**2*Bx**2 + 6*Ay**2*By**2 - 4*Ay*Bx**2*By - 4*Ay*By**3 + Bx**4 + 2*Bx**2*By**2 + By**4),
#          Z,
#          Z],
#         [(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2 - 2*(Ay - By)**2)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2)**2,
#          2*(Ax - Bx)*(Ay - By)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2)**2,
#          (-Ax**2 + 2*Ax*Bx + Ay**2 - 2*Ay*By - Bx**2 + By**2)/(Ax**4 - 4*Ax**3*Bx + 2*Ax**2*Ay**2 - 4*Ax**2*Ay*By + 6*Ax**2*Bx**2 + 2*Ax**2*By**2 - 4*Ax*Ay**2*Bx + 8*Ax*Ay*Bx*By - 4*Ax*Bx**3 - 4*Ax*Bx*By**2 + Ay**4 - 4*Ay**3*By + 2*Ay**2*Bx**2 + 6*Ay**2*By**2 - 4*Ay*Bx**2*By - 4*Ay*By**3 + Bx**4 + 2*Bx**2*By**2 + By**4),
#          2*(-Ax*Ay + Ax*By + Ay*Bx - Bx*By)/(Ax**4 - 4*Ax**3*Bx + 2*Ax**2*Ay**2 - 4*Ax**2*Ay*By + 6*Ax**2*Bx**2 + 2*Ax**2*By**2 - 4*Ax*Ay**2*Bx + 8*Ax*Ay*Bx*By - 4*Ax*Bx**3 - 4*Ax*Bx*By**2 + Ay**4 - 4*Ay**3*By + 2*Ay**2*Bx**2 + 6*Ay**2*By**2 - 4*Ay*Bx**2*By - 4*Ay*By**3 + Bx**4 + 2*Bx**2*By**2 + By**4),
#          Z,
#          Z],
#         [2*(Ax - Bx)*(Ay - By)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2)**2,
#          (Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2 - 2*(Ax - Bx)**2)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2)**2,
#          2*(((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2) - ((Ay - Cy)*((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy)) + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*(Ax - 2*Bx + Cx))*((Ay - Cy)*((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx)) - ((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))*(Ax - 2*Bx + Cx)))/(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2)**2,
#          (2*((Ax - Cx)*((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy)) - ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*(Ay - 2*By + Cy))*((Ay - Cy)*((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx)) - ((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))*(Ax - 2*Bx + Cx)) + ((Ax - Cx)*(Ax - 2*Bx + Cx) + (Ay - Cy)*(Ay - 2*By + Cy))*(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2))/(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2)**2,
#          -2*(Bx - Cx)*(By - Cy)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)**2,
#          (-Bx**2 + 2*Bx*Cx - By**2 + 2*By*Cy - Cx**2 - Cy**2 + 2*(Bx - Cx)**2)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)**2],
#         [(-Ax**2 + 2*Ax*Bx - Ay**2 + 2*Ay*By - Bx**2 - By**2 + 2*(Ay - By)**2)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2)**2,
#          -2*(Ax - Bx)*(Ay - By)/(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2)**2,
#          (2*((Ax - Cx)*((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx)) + ((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))*(Ay - 2*By + Cy))*((Ay - Cy)*((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy)) + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*(Ax - 2*Bx + Cx)) - ((Ax - Cx)*(Ax - 2*Bx + Cx) + (Ay - Cy)*(Ay - 2*By + Cy))*(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2))/(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2)**2,
#          2*(((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2) - ((Ax - Cx)*((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy)) - ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*(Ay - 2*By + Cy))*((Ax - Cx)*((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx)) + ((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))*(Ay - 2*By + Cy)))/(((Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy))**2 + ((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))**2)**2,
#          (Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2 - 2*(By - Cy)**2)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)**2,
#          2*(Bx - Cx)*(By - Cy)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)**2],
#         [Z,
#          Z,
#          2*(-Bx*By + Bx*Cy + By*Cx - Cx*Cy)/(Bx**4 - 4*Bx**3*Cx + 2*Bx**2*By**2 - 4*Bx**2*By*Cy + 6*Bx**2*Cx**2 + 2*Bx**2*Cy**2 - 4*Bx*By**2*Cx + 8*Bx*By*Cx*Cy - 4*Bx*Cx**3 - 4*Bx*Cx*Cy**2 + By**4 - 4*By**3*Cy + 2*By**2*Cx**2 + 6*By**2*Cy**2 - 4*By*Cx**2*Cy - 4*By*Cy**3 + Cx**4 + 2*Cx**2*Cy**2 + Cy**4),
#          (Bx**2 - 2*Bx*Cx - By**2 + 2*By*Cy + Cx**2 - Cy**2)/(Bx**4 - 4*Bx**3*Cx + 2*Bx**2*By**2 - 4*Bx**2*By*Cy + 6*Bx**2*Cx**2 + 2*Bx**2*Cy**2 - 4*Bx*By**2*Cx + 8*Bx*By*Cx*Cy - 4*Bx*Cx**3 - 4*Bx*Cx*Cy**2 + By**4 - 4*By**3*Cy + 2*By**2*Cx**2 + 6*By**2*Cy**2 - 4*By*Cx**2*Cy - 4*By*Cy**3 + Cx**4 + 2*Cx**2*Cy**2 + Cy**4),
#          2*(Bx - Cx)*(By - Cy)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)**2,
#          (Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2 - 2*(Bx - Cx)**2)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)**2],
#         [Z,
#          Z,
#          (Bx**2 - 2*Bx*Cx - By**2 + 2*By*Cy + Cx**2 - Cy**2)/(Bx**4 - 4*Bx**3*Cx + 2*Bx**2*By**2 - 4*Bx**2*By*Cy + 6*Bx**2*Cx**2 + 2*Bx**2*Cy**2 - 4*Bx*By**2*Cx + 8*Bx*By*Cx*Cy - 4*Bx*Cx**3 - 4*Bx*Cx*Cy**2 + By**4 - 4*By**3*Cy + 2*By**2*Cx**2 + 6*By**2*Cy**2 - 4*By*Cx**2*Cy - 4*By*Cy**3 + Cx**4 + 2*Cx**2*Cy**2 + Cy**4),
#          2*(Bx*By - Bx*Cy - By*Cx + Cx*Cy)/(Bx**4 - 4*Bx**3*Cx + 2*Bx**2*By**2 - 4*Bx**2*By*Cy + 6*Bx**2*Cx**2 + 2*Bx**2*Cy**2 - 4*Bx*By**2*Cx + 8*Bx*By*Cx*Cy - 4*Bx*Cx**3 - 4*Bx*Cx*Cy**2 + By**4 - 4*By**3*Cy + 2*By**2*Cx**2 + 6*By**2*Cy**2 - 4*By*Cx**2*Cy - 4*By*Cy**3 + Cx**4 + 2*Cx**2*Cy**2 + Cy**4),
#          (-Bx**2 + 2*Bx*Cx - By**2 + 2*By*Cy - Cx**2 - Cy**2 + 2*(By - Cy)**2)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)**2,
#          -2*(Bx - Cx)*(By - Cy)/(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2)**2]
#     ])
    
#     # Reorder to match coordinate ordering [xA, yA, xB, yB, xC, yC]
#     # Current ordering in H array: [row0=xA, row1=yA, row2=xB, row3=yB, row4=xC, row5=yC]
#     # Need to permute to correct order
#     # perm = np.array([0, 1, 4, 5, 2, 3])  # Identity for now - adjust if needed
#     # H = H[perm, :][:, perm]
    
#     return H.transpose(2, 0, 1)

