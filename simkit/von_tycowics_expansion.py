import numpy as np

def von_tycowicz_expansion(B, dim):
    """
    Expands a modal basis using von Tycowicz expansion.
    
    Args:
        B: dim*n x m matrix where each column is a mode
           B is ordered as [x1,y1,z1, x2,y2,z2, ...] for each mode
        dim: dimension of the problem (e.g. 2 for 2D, 3 for 3D)
        
    Returns:
        D: dim*n x (dim*dim)*m expanded basis
           D is ordered as [x1,y1,z1, x2,y2,z2, ...] for each mode
           First mode in B corresponds to first dim*dim columns in D
    """
    n = B.shape[0] // dim  # number of vertices
    m = B.shape[1]  # number of modes
    
    # Reshape B to separate components, maintaining the original ordering
    # B is ordered as [x1,y1,z1, x2,y2,z2, ...] for each mode
    B_reshaped = B.reshape(n, dim, m).transpose(1, 0, 2)
    
    # Create expanded basis by repeating and permuting components
    D = np.zeros((dim, n, dim * dim * m))
    
    # For each mode, create dim*dim new basis vectors
    for i in range(m):
        # Get the components for this mode
        mode = B_reshaped[:, :, i]  # shape: (dim, n)
        
        # Create all possible permutations of components
        for k in range(dim):
            for l in range(dim):
                # Calculate index in expanded basis
                # Ensure first mode goes to first dim*dim columns
                idx = i * dim * dim + k * dim + l
                # Copy component k to position l
                D[l, :, idx] = mode[k]
    
    # Reshape back to dim*n x (dim*dim)*m while maintaining [x1,y1,z1, x2,y2,z2, ...] ordering
    D = D.transpose(1, 0, 2).reshape(dim * n, dim * dim * m)
    
    return D



