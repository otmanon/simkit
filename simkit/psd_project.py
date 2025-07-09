import numpy as np


from .svd_rv import svd_rv


def psd_project(H, method='proj'):
    if H.ndim == 2:
        H = H[None, :, :]
    [s, U] = np.linalg.eigh(H)

    dim = H.shape[-1]

    if method == 'abs':
        s = np.abs(s)
    elif method == 'proj':
        s[s < 1e-6] = 1e-6

    Id = np.identity(dim)[None, ...]
    S = Id * s.reshape(-1, dim, 1)

    HbI = U @ S @ U.transpose(0, 2, 1)


    return HbI