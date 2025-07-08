import igl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


from .Curve import Curve
from .colors import *
class TriangleMesh():
    def __init__(self, X, T, facecolors=light_blue, edgecolors=light_blue, linewidths=1, outlinewidth=0.1):
        self.X = X
        self.T = T

        triangles = [X[face] for face in T]

        self.pc = PolyCollection(triangles, facecolors=facecolors, edgecolor=edgecolors, linewidth = linewidths, linewidths=linewidths)
        plt.gca().add_collection(self.pc)

        self.E = igl.boundary_facets(T)[0]
        self.E = self.E.reshape(-1, 2)
        self.outline = Curve(X, self.E, color=black, linewidth=outlinewidth)
        return

    def update_vertex_positions(self, X):
        self.X = X
        triangles = [X[face] for face in self.T]
        self.pc.set_verts(triangles)
        self.outline.update_vertex_positions(X)
        return

    def remove(self):
        self.pc.remove()
        self.outline.remove()
        plt.clf()
        return