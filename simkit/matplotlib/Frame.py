import igl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PolyCollection


from .colors import *
class Frame():
    def __init__(self, A,  linewidths=1, headwidth=3, headlength=2, headaxislength=2, pointsize=200, pointcolor=yellow, pointedgecolor=black, pointlinewidth=1, zorder=10):
        
        self.X0 = np.zeros((2, 2))
        self.V0 = np.identity(2)

        self.A = A
        self.X = A[:, None, -1] + self.X0
        self.V = A[:, :-1] @ self.V0

        self.qu = plt.quiver(self.X[:, 0], self.X[:, 1], self.V[:, 0], self.V[:, 1], 
                             color=[blue, red], linewidth=linewidths, headwidth=headwidth, headlength=headlength, headaxislength=headaxislength, zorder=zorder)
        
        self.sc = plt.scatter(self.X[:, 0], self.X[:, 1], facecolor=pointcolor, s=pointsize, linewidth=pointlinewidth, edgecolors=pointedgecolor, zorder=zorder)

        
        # plt.clf()
     
        return

    def update_frame(self, A):
        self.A = A
        self.X = A[:, None, -1] + self.X0
        self.V = A[:, :-1] @ self.V0

        self.qu.set_UVC(self.V[:, 0], self.V[:, 1])
        self.qu.set_offsets(self.X)

        self.sc.set_offsets((self.X))
        return

    def remove(self):
        self.qu.remove()
        self.sc.remove()
        plt.clf()
        return