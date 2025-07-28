import os
import igl
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from examples.modal_muscles.animation_viewers import animation_viewer_2D
import simkit as sk
from pathlib import Path
import polyscope as ps

height = 4.0
dim = 2

current_dir = os.path.dirname(__file__)
result_dir = os.path.join(current_dir, "results")
os.makedirs(result_dir, exist_ok=True)



import polyscope as ps
ps.init()
ps.set_ground_plane_mode("none")
p0s = [1.0, 2.0]
p1s = [ 1.0, 0.5]


for p0 in p0s:
    for p1 in p1s:
        p = np.array([p0, p1])
        [Ve, Ee] = sk.ellipse_outline(a=p[0], b=p[1], n=100)
        [X0, T, _, _, _] = igl.triangle.triangulate(Ve, Ee, flags='qa0.01')

        ps.register_curve_network('mesh', Ve, Ee, material="flat")
        ps.register_surface_mesh('mesh2', X0, T, edge_width=1.0, material="flat")
        
        ps.look_at([0, 0, 4], [0, 0, 0])        
        ps.screenshot(os.path.join(result_dir, "shape_space_" + str(p0) + "_" + str(p1)) + ".png", True)

        ps.show()
    
