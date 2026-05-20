import sympy as sp

import numpy as np

from .area_normals import area_normal_element, area_normal_gradient_element, area_normal_hessian_element

def triangle_areas(X, F):
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return triangle_area_element(x0, x1, x2)

def triangle_areas_gradient(X, F):
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return triangle_area_gradient_element(x0, x1, x2)

def triangle_areas_hessian(X, F):
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return triangle_area_hessian_element(x0, x1, x2)



def triangle_area_dnormal(n):
    area = np.linalg.norm(n, axis=1).reshape(-1, 1) / 2
    return area

def triangle_area_gradient_dnormal(an):
    norm = np.linalg.norm(an, axis=1).reshape(-1, 1) 
    
    darea_dnorm = an / (2 * norm)
    return darea_dnorm

def triangle_area_hessian_d2normal(an):
    
    an_norm = np.linalg.norm(an, axis=1).reshape(-1, 1)
    I = np.identity(3)[None, :, :]
    term_1 = I / (an_norm) 
    term_2 = (an[:, :, None] @ an[:, None, :]) / (an_norm[:, :, None]**3)
    
    combined_term = 0.5 * (term_1 - term_2)
    return combined_term


def triangle_area_element(x0, x1, x2):
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x1 - x0
    
    n = np.cross(e1, -e2)
    area = np.linalg.norm(n, axis=1) / 2
    
    return area
    
def triangle_area_gradient_element(x0, x1, x2):
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x1 - x0
    
    n = np.cross(e1, -e2)
    n_hat = n / np.linalg.norm(n, axis=1).reshape(-1, 1)
    da_dx0 = -0.5 * np.cross(n_hat, e0)
    da_dx1 = -0.5 * np.cross(n_hat, e1)
    da_dx2 = -0.5 * np.cross(n_hat, e2)
    
    da_dx = np.concatenate([da_dx0, da_dx1, da_dx2], axis=1)
    return da_dx


def triangle_area_hessian_element(x0, x1, x2):
    an = area_normal_element(x0, x1, x2)
    an_norm = np.linalg.norm(an, axis=1).reshape(-1, 1)
    dan_dx = area_normal_gradient_element(x0, x1, x2)
    
    I = np.identity(3)[None, :, :]
    term_1 = I / (an_norm[:, :, None]) 
    term_2 = (an[:, :, None] @ an[:, None, :]) / (an_norm[:, :, None]**3)
    
    combined_term = 0.5 * (term_1 - term_2)
    big_term_1 = dan_dx.transpose(0, 2, 1) @ combined_term @ dan_dx
    
    da_dn = triangle_area_gradient_dnormal(an)[:, :, None, None]
    d2an_dx2 = area_normal_hessian_element(x0, x1, x2)
    big_term_2 = (da_dn * d2an_dx2).sum(axis=1)
    
    d2a_dx2 = big_term_1 + big_term_2
    # d2A_dx2 = np.einsum('nik,nkl,nlj->nij', dan_dx, combined_term, dan_dx)
    return d2a_dx2





# x0 = np.random.randn(1, 3)
# x1 = np.random.randn(1, 3)
# x2 = np.random.randn(1, 3)

# normal = area_normal_element(x0, x1, x2)

# def energy(x):
#     x0 = x[:3].reshape(-1, 3)
#     x1 = x[3:6].reshape(-1, 3)
#     x2 = x[6:9].reshape(-1, 3)
#     return area_normal_element(x0, x1, x2)

# from simkit import gradient_cfd
# dn_dx_fd = gradient_cfd(energy, np.concatenate([x0, x1, x2], axis=1).flatten(), 1e-5)
# print(dn_dx_fd)
# dn_dx = area_normal_gradient_element(x0, x1, x2)
# print(np.linalg.norm(dn_dx - dn_dx_fd))



# x0 = np.random.randn(1, 3)
# x1 = np.random.randn(1, 3)
# x2 = np.random.randn(1, 3)

# def gradient(x):
#     x0 = x[:3].reshape(-1, 3)
#     x1 = x[3:6].reshape(-1, 3)
#     x2 = x[6:9].reshape(-1, 3)
#     return area_normal_gradient_element(x0, x1, x2)[0]

# from simkit import gradient_cfd
# dn_dx2_fd = gradient_cfd(gradient, np.concatenate([x0, x1, x2], axis=1).flatten(), 1e-5)
# dn_dx2 = area_normal_hessian_element(x0, x1, x2)
# print(dn_dx2 - dn_dx2_fd)
# print(np.linalg.norm(dn_dx2 - dn_dx2_fd))


# x0 = np.random.randn(1, 3)
# x1 = np.random.randn(1, 3)
# x2 = np.random.randn(1, 3)
# normal = area_normal_element(x0, x1, x2)
# def energy(x):
#     x0 = x[:3].reshape(-1, 3)
#     x1 = x[3:6].reshape(-1, 3)
#     x2 = x[6:9].reshape(-1, 3)
#     return area_normal_element(x0, x1, x2)

# from simkit import gradient_cfd
# dn_dx_fd = gradient_cfd(energy, np.concatenate([x0, x1, x2], axis=1).flatten(), 1e-5)
# print(dn_dx_fd)
# dn_dx = normal_gradient_element(x0, x1, x2)

# print(np.linalg.norm(dn_dx - dn_dx_fd))




# x0 = np.random.randn(1, 3)
# x1 = np.random.randn(1, 3)
# x2 = np.random.randn(1, 3)
# area = area_element(x0, x1, x2)
# def energy(x):
#     x0 = x[:3].reshape(-1, 3)
#     x1 = x[3:6].reshape(-1, 3)
#     x2 = x[6:9].reshape(-1, 3)
#     return area_element(x0, x1, x2)
# from simkit import gradient_cfd
# da_dx_fd = gradient_cfd(energy, np.concatenate([x0, x1, x2], axis=1).flatten(), 1e-5)
# da_dx = area_gradient_element(x0, x1, x2)
# print(da_dx - da_dx_fd)
# print(np.linalg.norm(da_dx - da_dx_fd))


# for i in range(10):
#     x0 = np.random.randn(1, 3)
#     x1 = np.random.randn(1, 3)
#     x2 = np.random.randn(1, 3)
#     def gradient(x):
#         x0 = x[:3].reshape(-1, 3)
#         x1 = x[3:6].reshape(-1, 3)
#         x2 = x[6:9].reshape(-1, 3)
#         return area_gradient_element(x0, x1, x2).flatten()

#     hess_fd = gradient_cfd(gradient, np.concatenate([x0[[0], :], x1[[0], :], x2[[0], :]], axis=1).flatten(), 1e-8)
#     hess= area_hessian_element(x0, x1, x2)

#     print(np.linalg.norm(hess - hess_fd))


