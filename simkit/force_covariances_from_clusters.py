import scipy as sp
import numpy as np

from simkit.massmatrix import massmatrix
from simkit.selection_matrix import selection_matrix
from simkit.vertex_to_simplex_adjacency import vertex_to_simplex_adjacency
from simkit.volume import volume

def force_covariances_from_clusters(X, T, labels, var=1.0):

    k = labels.max() + 1
    sigma_F_list = []
    label_occupancy = np.zeros((T.shape[0], k)) + 1e-4 # tiny reg
    vol = volume(X, T)
    Adj = vertex_to_simplex_adjacency(T, X.shape[0])
    for i in range(k):
        in_label = (labels == i)
        label_occupancy[in_label, i] = var
        
        face_variance = label_occupancy[:, i]*vol.reshape(-1)
        
        vertex_variance = (Adj @ face_variance)/T.shape[1]
        
        sigma_F = sp.sparse.diags(vertex_variance.flatten())
        sigma_F_list.append(sigma_F)
    
    #     # import polyscope as ps
    #     # ps.init()
    #     # mesh = ps.register_surface_mesh("mesh", X, T)
    #     # mesh.add_scalar_quantity("vertex_labels", vertex_labels, cmap='rainbow')
    #     # mesh.add_scalar_quantity("labels", labels, defined_on='faces', cmap='rainbow')
    #     # ps.show()
    # elif labels.shape[0] == X.shape[0]:
    #     vertex_labels = labels
    # else:
    #     raise ValueError("labels shape not recognized")
        
    # M = massmatrix(X=X, T=T)
    # Msqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
    
    # for i in range(k):
    #     distribution =  (labels == i)* 1.0 + (labels != i) * 1e-8
    #     sigma_F = Msqrt @ sp.sparse.diags(distribution) @ Msqrt
    
    #     sigma_F_list.append(sigma_F)
    
    return sigma_F_list        