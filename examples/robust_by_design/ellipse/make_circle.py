
import simkit as sk
import igl
import igl.triangle
import os
import numpy as np

ns = [10, 25, 100, 200]

for n in ns:
    [Ve, Ee] = sk.ellipse_outline(a=1, b=1, n=n)
    [X0, T, _, _, _] = igl.triangle.triangulate(Ve, Ee, flags='qa' + str(np.round(1/ (n), 3)))

    X0 = np.concatenate((X0, np.ones((X0.shape[0], 1))), axis=1)
    current_dir = os.path.dirname(__file__)
    igl.writeOBJ(current_dir + "/circle_" + str(n) + ".obj", X0, T)
    # igl.readOBJ(current_dir + "/circle_" + str(n) + ".obj")
    import polyscope as ps
    ps.init()

    ps.register_surface_mesh("mesh", X0, T)
    ps.show()