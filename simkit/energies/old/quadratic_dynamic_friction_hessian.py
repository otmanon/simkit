import scipy.sparse as sp

def quadratic_dynamic_friction_hessian(z, z_curr, tangents, S, k_dyn_friction, JJ=None, return_JJ=False):
    if JJ is None:
        dim = tangents.shape[1]
        
        Se = sp.kron(S, sp.identity(dim))
        v_t_dir = tangents
        D = sp.block_diag(v_t_dir[:, None, :])
        J = D @ Se
        JJ = J.T @ J
    
    if return_JJ:
        return  k_dyn_friction * (JJ), JJ
    else:
        return  k_dyn_friction * (JJ)