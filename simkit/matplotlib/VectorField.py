import igl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PolyCollection


from .colors import *
class VectorField():
    def __init__(self, X, V,  linewidths=1, headwidth=3, headlength=2, headaxislength=2, pointsize=200, pointcolor=yellow, color=black, pointedgecolor=black, pointlinewidth=1, zorder=10):
        
        self.X0 = X.copy()
        self.V0 = V.copy()

        self.X = X.copy()
        self.V = V.copy()

        self.qu = plt.quiver(self.X[:, 0], self.X[:, 1], self.V[:, 0], self.V[:, 1], 
                             color=color, linewidth=linewidths, headwidth=headwidth, headlength=headlength, headaxislength=headaxislength, zorder=zorder)
        
        # self.sc = plt.scatter(self.X[:, 0], self.X[:, 1], facecolor=pointcolor, s=pointsize, linewidth=pointlinewidth, edgecolors=pointedgecolor, zorder=zorder)

        
        # plt.clf()
     
        return

    def update_vector_field(self, X, V):

        self.X = X
        self.V = V

        self.qu.set_UVC(self.V[:, 0], self.V[:, 1])
        self.qu.set_offsets(self.X)

        return

    def remove(self):
        self.qu.remove()
        plt.clf()
        return