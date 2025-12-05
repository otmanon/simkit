import numpy as np
import scipy as sp

def backward_euler_rollout_matrix(num_timesteps):
    """
    Computes the rolled out backward euler matrix and boundary condition matrix assuming backward euler timestepping
    for num_timesteps timesteps over an entire trajectory.
    
    Given a number of timesteps, the resulting trajectory using backward euler timestepping is obtained by solving the linear system:
    T @ [x_0; x_1; ...; x_{num_timesteps-1}] = B @ [x_{-2}; x_{-1}]
    
    Where x_{-2} is two timesteps before the first timestep, x_{-1} is the first timestep.
    

    The boundary conditions are applied to the first and last two timesteps.
    """
    timesteps = np.arange(num_timesteps)
    inds = np.repeat(np.repeat(timesteps[:, None], 3, axis=1)[:, None], 3, axis=1)

    offset_i = np.array([[-2, -2, -2], [-1, -1, -1], [0, 0, 0]])
    offset_j = offset_i.T
    i = inds + offset_i
    j = inds + offset_j
    vals = np.array([[1/2, -1, 1/2],
                    [-1, 2, -1], 
                    [1/2, -1, 1/2]])
    vals = np.repeat(vals[None, :, :], num_timesteps, axis=0)
    i_f = i.flatten()
    j_f = j.flatten()
    v_f = vals.flatten()
    valid_inds_i = (i_f >= 0) & (i_f < num_timesteps)
    valid_inds_j = (j_f >= 0) & (j_f < num_timesteps)
    valid_inds = valid_inds_i & valid_inds_j
    vals = v_f[valid_inds]
    i = i_f[valid_inds]
    j = j_f[valid_inds]
    T = sp.sparse.csc_matrix((vals, (i, j)), shape=(num_timesteps, num_timesteps))
    B = sp.sparse.csc_matrix(([2, -0.5, -0.5], ([0, 0, 1], [1, 0, 1])), shape=(num_timesteps, 2))
    return T, B