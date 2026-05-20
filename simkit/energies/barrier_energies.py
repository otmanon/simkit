import numpy as np


# def friction_interpolant(y, epsilon_v, h):
#     f = y.copy()**3 
#     return f

# def friction_interpolant_gradient(y, epsilon_v, h):
#     g =  3* y.copy()**2 
#     return g

# def friction_interpolant_hessian(y, epsilon_v, h):
#     h =  6* y.copy() 
#     return h


# def friction_interpolant(y, epsilon_v, h):
#     threshold = h * epsilon_v
#     f = - y**3/(3 *threshold**2) +  y**2/threshold

#     return f

# def friction_interpolant_gradient(y, epsilon_v, h):
#     threshold = h * epsilon_v
#     g =  - y**2/(threshold**2) + 2 * y/threshold
#     return g

# def friction_interpolant_hessian(y, epsilon_v, h):
#     threshold = h * epsilon_v
#     h = - 2 * y / (threshold**2) + 2 * np.ones(y.shape)/ threshold
#     return h



def sticking_friction_interpolant(y, epsilon_v, h):
    f = 0.5* y**2
    return f

def sticking_friction_interpolant_gradient(y, epsilon_v, h):
    g = y.copy() 
    return g

def sticking_friction_interpolant_hessian(y, epsilon_v, h):
    
    hess = np.ones(y.shape) 
    return hess


def stick_slip_friction_interpolant(y, epsilon_v, h):
    threshold = h * epsilon_v
    
    below_threshold = (y < threshold).flatten()
    
    f = y.copy()
    yl = y[below_threshold]
    f[below_threshold] = - yl**3/(3 *threshold**2) +  yl**2/threshold + threshold/3
    return f

def stick_slip_friction_interpolant_gradient(y, epsilon_v, h):
    
    threshold = h * epsilon_v
    
    below_threshold = (y < threshold).flatten()
    
    yl = y[below_threshold]
    g = np.ones(y.shape)
    g[below_threshold] = - yl**2/(threshold**2) + 2 * yl/threshold

    return g

def stick_slip_friction_interpolant_hessian(y, epsilon_v, h):
    threshold = h * epsilon_v
    below_threshold = (y < threshold).flatten()
    yl = y[below_threshold]
    
    h = np.zeros((y.shape[0], 1))
    h[below_threshold] = - 2 * yl / (threshold**2) + 2 * np.ones(yl.shape)/ threshold
    # h[below_threshold] = - 2 * yl/threshold**2 + 2/threshold
    return h
    

def quadratic_barrier_energy(d, d_hat):
    d = d.reshape(-1, 1)
    energy_densities = np.zeros(d.shape)
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    e = (dl - d_hat)**2
    energy_densities[less_than_dhat] = e
    return energy_densities

def quadratic_barrier_gradient(d, d_hat):
    d = d.reshape(-1, 1)
    grads = np.zeros((d.shape[0], 1))
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    g = 2*(dl - d_hat)
    grads[less_than_dhat] = g

    return grads

def quadratic_barrier_hessian(d, d_hat):
    d = d.reshape(-1, 1)
    hess = np.zeros((d.shape[0], 1))
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    h = 2 * np.ones((dl.shape[0], ))
    hess[less_than_dhat] = h
    return hess


def cubic_barrier_energy(d, d_hat):
    d = d.reshape(-1, 1)
    energy_densities = np.zeros(d.shape)
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    e = (np.abs(dl - d_hat))**3
    energy_densities[less_than_dhat] = e
    return energy_densities

def cubic_barrier_gradient(d, d_hat):
    d = d.reshape(-1, 1)
    grads = np.zeros((d.shape[0], 1))
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    g = -3*(np.abs(dl - d_hat))**2
    grads[less_than_dhat] = g
    return grads

def cubic_barrier_hessian(d, d_hat):
    d = d.reshape(-1, 1)
    hess = np.zeros((d.shape[0], 1))
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    h = 6*(np.abs(dl - d_hat))
    hess[less_than_dhat] = h
    return hess


def ipc_barrier_energy( d, d_hat):
    d = d.reshape(-1, 1)
    energy_densities = np.zeros(d.shape)
    
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    e = - (dl - d_hat)**2 * np.log(dl / d_hat)
    energy_densities[less_than_dhat] = e
    return energy_densities

def ipc_barrier_gradient(d, d_hat):
    d = d.reshape(-1, 1)
    grads = np.zeros((d.shape[0], 1))
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    
    
    g = - 2*(dl - d_hat) * np.log(dl / d_hat) + \
        - (dl - d_hat)**2 *(1.0/dl) 
        
    grads[less_than_dhat] = g
    return grads

def ipc_barrier_hessian(d, d_hat):
    d = d.reshape(-1, 1)
    hess = np.zeros((d.shape[0], 1))
    less_than_dhat = d < d_hat
    dl = d[less_than_dhat]
    
    h = - 2 * np.log(dl / d_hat)  \
        - 4*(dl - d_hat) * (1/dl)  \
        + (dl - d_hat)**2 *(1/dl)**2
        
    hess[less_than_dhat] = h
    return hess
    
    