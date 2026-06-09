

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PolyCollection


from .colors import *
class PointCloud():
    def __init__(self, X,  size=200, color=yellow, edgecolor=black, linewidth=1, zorder=10):

        self.X = X
        self.sc = plt.scatter(self.X[:, 0]*0, self.X[:, 1]*0, facecolor=color, s=size, linewidth=linewidth, edgecolors=edgecolor, zorder=zorder)
        self.sc.set_offsets((X))
        return

    def update_vertex_positions(self, X):
        self.X = X
        self.sc.set_offsets(X)
        return

    def remove(self):
        self.sc.remove()
        plt.clf()
        return