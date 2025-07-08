import simkit as sk
import numpy as np
import matplotlib.pyplot as plt

def animation_viewer_2D(Zs, B, X, T, cI, ground_height):
    dim = X.shape[1]
    num_timesteps = Zs.shape[1]
    com0, SB = sk.subspace_com(Zs[:, 0], B, X, T, return_SB=True)
    R, GAJB = sk.subspace_rotation(Zs[:, 0], B, X, T, return_GAJB=True)

    forward = np.array([[1, 0]])
    [fig, ax] = plt.subplots(dpi=100, figsize=(5, 5))
    plt.ion()
    pc2 = sk.matplotlib.PointCloud(com0.reshape(-1, 2), size=100, color='red')
    plt.axline((0, ground_height), slope=0, color='black', linewidth=4, zorder=-1)
    plt.axis('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)


    mesh = sk.matplotlib.TriangleMesh(X, T, edgecolors=sk.matplotlib.gray, linewidths=0.1,  outlinewidth=2)
    pc = sk.matplotlib.PointCloud(X[cI, :], size=10)
    vf = sk.matplotlib.VectorField(com0, forward)

    Us = B @ Zs
    for i in range(num_timesteps):    
        U = Us[:, i].reshape(-1, dim)    

        mesh.update_vertex_positions(U)
        pc.update_vertex_positions(U[cI, :])
        com = sk.subspace_com(Zs[:, i], B, X, T, SB=SB)
        R = sk.subspace_rotation(Zs[:, i], B, X, T, GAJB=GAJB)
        vf.update_vector_field(com, (R @ forward.T).T)
        pc2.update_vertex_positions(com.reshape(-1, 2))

        com_disp = (com - com0).flatten()
        ax.set_xlim(-5 + com_disp[0], 5 + com_disp[0])
        ax.set_ylim(-5 + com_disp[1], 5 + com_disp[1])
        plt.pause(0.001)
    