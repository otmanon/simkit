from simkit.dirichlet_penalty import dirichlet_penalty

from AdaptiveElasticFEMSim import *

class AdaptiveClickFEMSim(AdaptiveElasticFEMSim):

    def __init__(self, k, X : np.ndarray, T : np.ndarray, B : np.ndarray | sp.sparse.csc_matrix =None, cI : np.ndarray=None,  cW : np.ndarray=None, sim_params :ElasticFEMSimParams =ElasticFEMSimParams(), q=None):
        """
        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        super().__init__( X, T, B, cI, cW, sim_params, q)

        # assert(x0 is not None) # constraints are formulated to act on displacements.
        
        self.k = k
        self.u_handle = np.zeros((0, X.shape[1]))
        self.x0_handle = np.zeros((0, X.shape[1]))
        self.handle_indices = np.zeros((0), dtype=int)
        self.handle_indices_expanded = np.zeros((0), dtype=int)
        
        self.Q_handle = None
        self.b_handle = None
        self.SGamma = None
        self.S = None
        
        
        self.BQB_handle = None
        self.Bb_handle = None
        self.BSGamma = None
        self.SB = None
        self.num_handles = 0
        
        return

    def add_handle(ss, new_ind, handle_pos):
        ss.handle_indices = np.concatenate((ss.handle_indices, new_ind), axis=0)
        new_ind_expanded = new_ind * ss.dim + np.arange(ss.dim)
        ss.handle_indices_expanded = np.concatenate((ss.handle_indices_expanded, new_ind_expanded))
        
        x0_handle = ss.X[new_ind]
        u_handle = handle_pos - x0_handle
        ss.x0_handle = np.concatenate((ss.x0_handle, x0_handle), axis=0)
        ss.u_handle = np.concatenate((ss.u_handle, u_handle), axis=0)
        
        ss.S = selection_matrix(ss.handle_indices_expanded, ss.X.shape[0] * ss.dim)
        ss.SB =  ss.S @ ss.B
        [ss.Q_handle, ss.b_handle, ss.SGamma] = dirichlet_penalty(ss.handle_indices,
                                                                  ss.u_handle, ss.X.shape[0],  ss.k, return_SGamma=True)
        
        ss.BQB_handle = ss.B.T @ ss.Q_handle @ ss.B
        ss.BSGamma = ss.B.T @ ss.SGamma
        ss.Bb_handle =  - ss.BSGamma @ ss.u_handle.reshape(-1, 1)
        
        ss.num_handles += 1
        return
    
    def update_handle_position(ss, handle_index, handle_pos):
        ss.u_handle[handle_index] = handle_pos - ss.x0_handle[handle_index]
        ss.b_handle = - ss.SGamma @ ss.u_handle.reshape(-1, 1)
        ss.Bb_handle = - ss.BSGamma @ ss.u_handle.reshape(-1, 1)
        return
    
    def remove_handle(ss, handle_index):
        ss.u_handle = np.delete(ss.u_handle, handle_index, axis=0)
        ss.x0_handle = np.delete(ss.x0_handle, handle_index, axis=0)
        ss.handle_indices = np.delete(ss.handle_indices, handle_index, axis=0)
        ss.handle_indices_expanded = np.delete(ss.handle_indices_expanded, handle_index * ss.dim + np.arange(ss.dim), axis=0)
        
        ss.S = selection_matrix(ss.handle_indices_expanded, ss.X.shape[0] * ss.dim)
        ss.SB =  ss.S @ ss.B
        [ss.Q_handle, ss.b_handle, ss.SGamma] = dirichlet_penalty(ss.handle_indices,
                                                                  ss.u_handle, ss.X.shape[0],  ss.k, return_SGamma=True)
        
        ss.BQB_handle = ss.B.T @ ss.Q_handle @ ss.B
        ss.BSGamma = ss.B.T @ ss.SGamma
        ss.Bb_handle =  - ss.BSGamma @ ss.u_handle.reshape(-1, 1)
        
        ss.num_handles -= 1
        
    def update_subspace(ss, B, cI, cW):
        # ss.SB = ss.Se @ B

        super().update_subspace(B, cI, cW)
        
        ss.BQB_handle = ss.B.T @ ss.Q_handle @ ss.B
        ss.BSGamma = ss.B.T @ ss.SGamma
        ss.Bb_handle =  - ss.BSGamma @ ss.u_handle.reshape(-1, 1)
        
        # ss.BQB_handle = ss.B.T @ ss.Q_handle @ ss.B
        # ss.BSGamma = ss.B.T @ ss.SGamma
        # ss.Bb_handle =  - ss.BSGamma @ ss.u_handle.reshape(-1, 1)
        
        
        ss.b_handle = - ss.SGamma @ ss.u_handle.reshape(-1, 1)
        ss.Bb_handle = - ss.BSGamma @ ss.u_handle.reshape(-1, 1)
        
        ss.SB = ss.S @ ss.B
        return
    
    
    def step(self, z : np.ndarray, z_dot : np.ndarray):
        z_next = super().step(z, z_dot, BQB_ext=self.BQB_handle, Bb_ext=self.Bb_handle)
        return z_next
    

    def query_handle_force(ss, z):
        
        u_curr = ss.SB @ z 
        f = ss.SGamma @ ( u_curr - ss.u_handle.reshape(-1, 1) )
        return f


    def query_full_handle_force(ss, z):
        u_curr = ss.SB @ z 
        f = ss.SGamma @ ( u_curr - ss.u_handle.reshape(-1, 1) )
        return f
