
import numpy as np

from .pairwise_distance import pairwise_distance


def create_selection(name, X, t):
    if name == "center":
        pinned, pinnedI = center_indices(X, t)
    elif name == "back_z":
        pinned, pinnedI = back_z_indices(X, t)
    else:
        raise ValueError("Unknown pinning type")
    return pinned, pinnedI

def back_z_indices(X, t):

    diff = X[:, 2].max() - X[:, 2].min()

    pinned= X[:, 2] < X[:, 2].min() + diff * t
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI
 
def center_indices(X, t):
    diff = np.max(X.max(axis=0) - X.min(axis=0))

    center = X.mean(axis=0)
    pinned = np.linalg.norm(X - center, axis=1) < diff * t
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI

def top_indices(X, t):
    diff = X[:, 1].max() - X[:, 1].min()

    pinned= X[:, 1] > X[:, 1].max() - diff * t
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI

def bottom_indices(X, t):   
    diff = X[:, 1].max() - X[:, 1].min()

    pinned= X[:, 1] < X[:, 1].min() + diff * t
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI

def center_top_indices(X, t):
    diff = np.max(X.max(axis=0) - X.min(axis=0))

    center = X.mean(axis=0)
    pinned = np.abs(X[:, 0] - center[0]) < diff * t
    pinned = np.logical_and(pinned , (X[:, 1] > (X[:, 1].max() - diff * t)))
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI