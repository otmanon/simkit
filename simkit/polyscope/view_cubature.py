import numpy as np

import polyscope as ps

from ..average_onto_simplex import average_onto_simplex

def view_cubature(X, T, cI, cW=None, labels=None):
    ps.init()


    mesh = ps.register_surface_mesh("mesh", X, T)


    bc = average_onto_simplex(X, T)
    points = ps.register_point_cloud("cubature", bc[cI, :], radius=0.05)
    
    if labels is not None:
        mesh.add_scalar_quantity("labels", labels, defined_on='faces', cmap='rainbow', enabled=True)

    if cW is not None:
        points.add_scalar_quantity("cW", cW.flatten())

        # Set the quantity as the point size
        points.set_point_radius_quantity("cW")


    
    ps.show()