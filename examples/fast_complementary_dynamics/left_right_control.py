import os
import igl
import polyscope as ps
import numpy as np
import scipy as sp
import gpytoolbox as gpy # TODO: do we want gpy as a dependency?
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

from simkit.sims.elastic.FastCoDySim import FastCoDySim, FastCoDySimParams
from simkit.lbs_jacobian import lbs_jacobian
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.spectral_cubature import spectral_cubature
from simkit.diffuse_field import diffuse_field
from simkit.project_into_subspace import project_into_subspace
from simkit.massmatrix import massmatrix
from simkit.lbs_weight_space_constraint import lbs_weight_space_constraint
from simkit.normalize_and_center import normalize_and_center
from simkit.orthonormalize import orthonormalize

from simkit.filesystem.video_from_image_dir import video_from_image_dir
from simkit.filesystem.mp4_to_gif     import mp4_to_gif
from simkit.common_selections import *

from simkit.matplotlib.TriangleMesh import TriangleMesh, light_red, purple
from simkit.matplotlib.Frame import Frame
from simkit.matplotlib.PointCloud import PointCloud

from simkit.dirichlet_penalty import dirichlet_penalty


# Load a mesh in (X,T) from a file
dir = os.path.dirname(os.path.realpath(__file__))

class PinLeftRightWorld():
    def __init__(self, X, T, bI=None):
        dim = X.shape[1]
        M = massmatrix(X, T)
        Mv = sp.sparse.kron(M, sp.sparse.identity(dim))

        # rig space 
        J = X.reshape(-1, 1)

        # simulation subspace
        [W,_E,  B] = skinning_eigenmodes(X, T, 10)
        B = orthonormalize(B, M=Mv)

        [_cI, _cW, l] = spectral_cubature(X, T, W, 20, return_labels=True)

        if bI is None:
            bI = center_indices(X, 0.1)[1]

        self.bI = bI
        bc = np.zeros((self.bI.shape[0], 2))

        self.gamma = 1e12
        Q0, b0= dirichlet_penalty(self.bI, bc, X.shape[0], self.gamma)
        BQB = B.T @ Q0 @ B

        sim_params = FastCoDySimParams(ym = 1e5, rho=1e3, Q0=BQB)
        sim_params.solver_p.max_iter = 10
        sim = FastCoDySim(X, T,  J, B, l, sim_params)

        self.X = X
        self.T = T
        self.sim = sim
        self.J = J
        self.B = B


    def simulate_periodic_x(self, num_timesteps=500, period=100, a=1.0):

        [z, p, z_dot, p_dot, p_next] = self.sim.rest_state()
        p_next = p.copy()
        period = period
        
        Zs = np.zeros((z.shape[0], num_timesteps + 1))
        Ps = np.zeros((p.shape[0], num_timesteps + 1))
        Zs[:, 0] = z.flatten()
        Ps[:, 0] = p.flatten()

        for i in range(num_timesteps):
            
            bc = np.zeros((self.bI.shape[0], 2))
            bc[:, 0] = a * np.sin(2 * np.pi * i / period)
            b = dirichlet_penalty(self.bI, bc, X.shape[0], self.gamma, only_b=True)[0]

            z_next = self.sim.step(z, p, z_dot, p_dot, p_next, b_ext=self.B.T @ b)

            z_dot = (z_next - z) / self.sim.params.h
            p_dot = (p_next - p ) / self.sim.params.h

            z = z_next.copy()
            
            Zs[:, i+1] = z.flatten()
            Ps[:, i+1] = p.flatten()
        
        return Zs, Ps
    
    def render(self, Zs, Ps, path=None, save_tmp=False):
        Xs = self.B @ Zs +self.J @ Ps
        T = self.T
        X = Xs[:, 0].reshape(-1, 2)
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.ion()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.axis('tight')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.margins(x=0, y=0)
        purple = np.array([201,148,199])/255
        mesh = TriangleMesh(X, T, linewidths=1,  outlinewidth=1, facecolors=purple, edgecolors=purple)
        pc = PointCloud(X[self.bI, :], color=[0.75, 0.75, 0.75, 1], size=50, linewidth=0.5)

        if path is not None:
            # get directory from path
            dir = os.path.join(os.path.dirname(path), Path(path).stem) 
            os.makedirs(dir, exist_ok=True)

        assert(Zs.shape[1] == Ps.shape[1])

        for i in range(Zs.shape[1]):
            X = Xs[:, i].reshape(-1, 2)

            mesh.update_vertex_positions(X)
            pc.update_vertex_positions(X[self.bI, :])
            plt.pause(0.0001)

            if path is not None:
                plt.savefig(os.path.join(dir, f"{i:04d}.png"))
  
        plt.clf()
        if path is not None:
            video_from_image_dir(dir, path, fps=30)
            # replace path ending with gif
            path_gif = path.replace(".mp4", ".gif")
            mp4_to_gif(path, path_gif)

        if not save_tmp:
            if path is not None:
                # delete temporary directory
                shutil.rmtree(dir)

class CoDyLeftRightWorld():
    def __init__(self, X, T):
        dim = X.shape[1]
        M = massmatrix(X, T)
        Mv = sp.sparse.kron(M, sp.sparse.identity(dim))

        # rig space 
        J = lbs_jacobian(X, np.ones((X.shape[0], 1)))

        # momentum leaking matrix
        bI = np.unique(igl.boundary_facets(T)[0])
        d = diffuse_field(X, T, bI, np.ones((bI.shape[0], 1)), dt=1, normalize=True)
        # d = (d - d.min()) / (d.max() - d.min())
        D =  sp.sparse.diags((1 - d).flatten())
        De = sp.sparse.kron(D, sp.sparse.identity(dim))

        # orthogonality constraint
        O = De @ Mv @ J
        # weight space orthogonality constraint
        Aeq = lbs_weight_space_constraint(X, T, O.T)
        
        # simulation subspace
        [W,_E,  B] = skinning_eigenmodes(X, T, 10, Aeq=Aeq)
        B = orthonormalize(B, M=Mv)

        [_cI, _cW, l] = spectral_cubature(X, T, W, 20, return_labels=True)
        sim_params = FastCoDySimParams(ym = 1e5, rho=1e3)
        sim_params.solver_p.max_iter = 10
        sim = FastCoDySim(X, T,  J, B, l, sim_params)

        self.X = X
        self.T = T
        self.sim = sim
        self.J = J
        self.B = B

        return
    
    def simulate_periodic_x(self, num_timesteps=500, period=100, a=1.0):

        [z, p, z_dot, p_dot, p_next] = self.sim.rest_state()
        p_next = p.copy()
        period = period
        
        Zs = np.zeros((z.shape[0], num_timesteps + 1))
        Ps = np.zeros((p.shape[0], num_timesteps + 1))
        Zs[:, 0] = z.flatten()
        Ps[:, 0] = p.flatten()

        for i in range(num_timesteps):
            p_next[-2] = a* np.sin(2 * np.pi * i / period)    
            z_next = self.sim.step(z, p, z_dot, p_dot, p_next)

            z_dot = (z_next - z) / self.sim.params.h
            p_dot = (p_next - p ) / self.sim.params.h

            p = p_next.copy()
            z = z_next.copy()

            Zs[:, i+1] = z.flatten()
            Ps[:, i+1] = p.flatten()
        
        return Zs, Ps
    

    
    def render(self, Zs, Ps, path=None, save_tmp=False):
        Xs = self.B @ Zs +self.J @ Ps
        T = self.T
        X = Xs[:, 0].reshape(-1, 2)
        P = Ps[:, 0].reshape(3, 2).T
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.ion()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.axis('tight')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.margins(x=0, y=0)
        mesh = TriangleMesh(X, T, linewidths=1,  outlinewidth=1)
        frame = Frame(P)

        if path is not None:
            # get directory from path
            dir = os.path.join(os.path.dirname(path), Path(path).stem) 
            os.makedirs(dir, exist_ok=True)

        assert(Zs.shape[1] == Ps.shape[1])

        for i in range(Zs.shape[1]):
            X = Xs[:, i].reshape(-1, 2)
            P = Ps[:, i].reshape(3, 2).T
            mesh.update_vertex_positions(X)
            frame.update_frame(P)
            plt.pause(0.0001)

            if path is not None:
                plt.savefig(os.path.join(dir, f"{i:04d}.png"))
  
        plt.clf()
        if path is not None:
            video_from_image_dir(dir, path, fps=30)
            # replace path ending with gif
            path_gif = path.replace(".mp4", ".gif")
            mp4_to_gif(path, path_gif)

        if not save_tmp:
            if path is not None:
                # delete temporary directory
                shutil.rmtree(dir)


    def render_rig(self, Ps, path=None, save_tmp=False):
        Xs = self.J @ Ps
        T = self.T
        X = Xs[:, 0].reshape(-1, 2)
        P = Ps[:, 0].reshape(3, 2).T
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.ion()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.axis('tight')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.margins(x=0, y=0)
        mesh = TriangleMesh(X, T, facecolors=light_red, edgecolors=light_red, linewidths=1,  outlinewidth=1)
        frame = Frame(P)
        if path is not None:
            # get directory from path
            dir = os.path.join(os.path.dirname(path), Path(path).stem) 
            os.makedirs(dir, exist_ok=True)

        assert(Zs.shape[1] == Ps.shape[1])

        for i in range(Zs.shape[1]):
            X = Xs[:, i].reshape(-1, 2)
            P = Ps[:, i].reshape(3, 2).T
            mesh.update_vertex_positions(X)
            frame.update_frame(P)
            plt.pause(0.0001)

            if path is not None:
                plt.savefig(os.path.join(dir, f"{i:04d}.png"))
  
        plt.clf()
        if path is not None:
            video_from_image_dir(dir, path, fps=30)
            # replace path ending with gif
            path_gif = path.replace(".mp4", ".gif")
            mp4_to_gif(path, path_gif)

        if not save_tmp:
            if path is not None:
                # delete temporary directory
                shutil.rmtree(dir)


# [X, T] = gpy.regular_square_mesh(20, 20)
# X = normalize_and_center(X)

# world_cd = CoDyLeftRightWorld(X, T)
# video_path = os.path.join(dir, "results", "cody_left_right.mp4")
# [Zs, Ps] = world_cd.simulate_periodic_x(num_timesteps=300, period=150, a=3)
# world_cd.render( Zs, Ps, path=video_path)

# video_path = os.path.join(dir, "results", "rig_left_right.mp4")
# world_cd.render_rig( Ps, path=video_path)

# bI = center_indices(X, 0.1)[1]
# world_pin = PinLeftRightWorld(X, T)
# video_path = os.path.join(dir, "results", "pin_0.1_left_right.mp4")
# [Zs, Ps] = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
# world_pin.render( Zs, Ps, path=video_path)

# bI = center_indices(X, 0.4)[1]
# world_pin = PinLeftRightWorld(X, T, bI=bI)
# video_path = os.path.join(dir, "results", "pin_0.4_left_right.mp4")
# [Zs, Ps] = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
# world_pin.render( Zs, Ps, path=video_path)



[X, T] = gpy.regular_square_mesh(40, 5)
X[:, 1] *= 0.1
Y = X.copy()
X[:, 0] = -Y[:, 1]
X[:, 1] = Y[:, 0]
X = normalize_and_center(X)

world_cd = CoDyLeftRightWorld(X, T)
video_path = os.path.join(dir, "results", "beam_cody_left_right.mp4")
[Zs, Ps] = world_cd.simulate_periodic_x(num_timesteps=300, period=150, a=3)
world_cd.render( Zs, Ps, path=video_path)
# video_path = os.path.join(dir, "results", "beam_rig_left_right.mp4")
# world_cd.render_rig( Ps, path=video_path)


bI = center_indices(X, 0.1)[1]
world_pin = PinLeftRightWorld(X, T)
video_path = os.path.join(dir, "results", "beam_pin_0.1_left_right.mp4")
[Zs, Ps] = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
world_pin.render( Zs, Ps, path=video_path)

# bI = center_indices(X, 0.4)[1]
# world_pin = PinLeftRightWorld(X, T, bI=bI)
# video_path = os.path.join(dir, "results", "beam_pin_0.4_left_right.mp4")
# [Zs, Ps] = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
# world_pin.render( Zs, Ps, path=video_path)


# [X, _, _, T, _, _] = igl.read_obj(dir + "/../../data/2d/cthulu/cthulu.obj")
# X = X[:, :2]
# X = normalize_and_center(X)
# X*=1.25
# bI = center_indices(X, 0.1)[1]
# world_pin = PinLeftRightWorld(X, T, bI=bI)
# video_path = os.path.join(dir, "results", "cthulu_center_pin_0.1_left_right.mp4")
# [Zs, Ps] = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
# world_pin.render( Zs, Ps, path=video_path)


# [X, _, _, T, _, _] = igl.read_obj(dir + "/../../data/2d/cthulu/cthulu.obj")
# X = X[:, :2]
# X = normalize_and_center(X)
# X*=1.25
# bI = center_top_indices(X, 0.1)[1]
# world_pin = PinLeftRightWorld(X, T, bI=bI)
# video_path = os.path.join(dir, "results", "cthulu_center_top_pin_0.1_left_right.mp4")
# [Zs, Ps] = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
# world_pin.render( Zs, Ps, path=video_path)



# [X, _, _, T, _, _] = igl.read_obj(dir + "/../../data/2d/cthulu/cthulu.obj")
# X = X[:, :2]
# X = normalize_and_center(X)
# X*=1.25
# bI = top_indices(X, 0.1)[1]
# world_pin = PinLeftRightWorld(X, T, bI=bI)
# video_path = os.path.join(dir, "results", "cthulu_top_pin_0.1_left_right.mp4")
# [Zs, Ps] = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
# world_pin.render( Zs, Ps, path=video_path)




# [X, _, _, T, _, _] = igl.read_obj(dir + "/../../data/2d/cthulu/cthulu.obj")
# X = X[:, :2]
# X = normalize_and_center(X)
# X*=1.25
# world_pin = CoDyLeftRightWorld(X, T)
# video_path = os.path.join(dir, "results", "cthulu_cd_left_right.mp4")
# [Zs, Ps] = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
# world_pin.render( Zs, Ps, path=video_path)
