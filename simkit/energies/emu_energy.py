import numpy as np


def emu_energy_F(F, d, a, vol):
    """
    F - t x d x d
    d - t x d
    a - t x 1
    """
    FF = F.transpose(0, 2, 1) @ F
    de = d[:, :, None].copy()
    dFFd = de.transpose(0, 2, 1) @ FF @ de
    energy = np.sum(vol.reshape(-1, 1,1) * a.reshape(-1, 1, 1) * dFFd) * 0.5
    return energy