from .area_normals import area_normal_element, area_normal_gradient_element, area_normal_hessian_element
import numpy as np
def normals(X, F):
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return normal_element(x0, x1, x2)


def normals_gradient(X, F):
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return normal_gradient_element(x0, x1, x2)


def normal_element(x0, x1, x2):
    area_normal = area_normal_element(x0, x1, x2)
    double_area = np.linalg.norm(area_normal, axis=1).reshape(-1, 1)
    normal = area_normal / double_area
    return normal
    
    
def normal_gradient_element(x0, x1, x2):
    
    area_normal = area_normal_element(x0, x1, x2)
    double_area = np.linalg.norm(area_normal, axis=1).reshape(-1, 1)

    dan_dx = area_normal_gradient_element(x0, x1, x2)
    
    I = np.identity(3)[None, :, :]
    term_1 = I / double_area[:, :, None]
    term_2 = area_normal[:, :, None] @ area_normal[:, None, :] / double_area[:, :, None]**3
    dn_dan = term_1 - term_2
    
    dn_dx = dn_dan @ dan_dx
    
    return dn_dx