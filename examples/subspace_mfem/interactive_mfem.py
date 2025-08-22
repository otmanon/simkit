import numpy as np
import sys
sys.path.append("../../")

import igl
import polyscope as ps
import polyscope.imgui as psim
import scipy as sp
import simkit as sk
import os

from utils import *
from config import *


def interactive_click_demo(sim, bI):
    import polyscope as ps

    if isinstance(bI, str):
        bI = np.load(bI).reshape((-1))

    s = None
    l = None
    if isinstance(sim, sk.sims.elastic.ElasticMFEMSim):
        z, s, l, z_dot = sim.rest_state()
    elif isinstance(sim, sk.sims.elastic.ElasticFEMSim):
        z, z_dot = sim.rest_state()

    ps.set_ground_plane_mode("none")
    
    y = (sim.X - sim.q.reshape(-1, sim.dim))[bI, :]
    [Q_pin, b_pin] = sk.dirichlet_penalty(bI, y, sim.X.shape[0],  k_pin)
    BQB_pin = sim.B.T @ Q_pin @ sim.B
    Bb_pin = sim.B.T @ b_pin
    Bb_gravity = -sim.B.T @ sk.gravity_force(sim.X, sim.T, a=ag, rho=1e3).reshape(-1, 1)
           
    BQB_ext = BQB_pin
    Bb_ext = Bb_pin + Bb_gravity 
    
    BSGamma = None
    d = None
    pc = None
    
    Y = sim.X.copy()
    clickedInd = None
    ps.init()
    
    if sim.T.shape[1] == 3:
        mesh = ps.register_surface_mesh("mesh", sim.X, sim.T)
    else:
        F, _, _ = igl.boundary_facets(sim.T)
        mesh = ps.register_surface_mesh("mesh", sim.X, F)
        
    def callback():
        nonlocal z, s, l, z_dot, BQB_ext, Bb_ext, d, pc, BSGamma, Y, clickedInd
    
        # get window pos
        win_pos = psim.GetMousePos()    
        view = ps.get_camera_view_matrix()
        camera_pos = np.linalg.inv(view)[:3, 3]  # get camera position

        # if right mouse button is clicked, place a point on the mesh.
        if psim.IsMouseClicked(1):     
            pos = ps.screen_coords_to_world_position(win_pos).reshape(-1, 3)    # use polyscope to find intersection into scene
            d = np.linalg.norm(pos - camera_pos)    # remember depth for future dragging
            pc = ps.register_point_cloud("clicked", pos, radius=0.01)   #vis
            clickedInd = np.array([np.argmin(sk.pairwise_distance(Y, pos[:, :dim]))])
            
            # update Bb_ext
            P0 = pos[:, :dim].copy()
            y = P0 - (sim.X)[clickedInd, :] 
            [Q_handle, b_handle, SGamma] = sk.dirichlet_penalty(clickedInd, y, sim.X.shape[0], 
                                                                k_pin, return_SGamma=True)
            BQB_ext = sim.B.T @ Q_handle @ sim.B + BQB_pin 
            BSGamma = -sim.B.T @ SGamma 
            Bb_ext = BSGamma @ y.reshape(-1, 1) + Bb_pin + Bb_gravity
            

        # if point being moved exists, and space is being held down, move the point by dragging mouse around
        if pc is not None and psim.IsKeyDown(psim.GetKeyIndex(psim.ImGuiKey_Space)): 
            ray = ps.screen_coords_to_world_ray(win_pos)   # shoot ray from camera into scene
            P = (camera_pos + d * ray).reshape(-1, 3)[:, :dim   ]   # get final position
            pc.update_point_positions(P)
            
            P0 = P.copy()
            y =  P0 - sim.X[clickedInd, :]
            Bb_handle = BSGamma @ y.reshape(-1, 1)
            Bb_ext = Bb_handle + Bb_pin + Bb_gravity
            
                    

        if isinstance(sim, sk.sims.elastic.ElasticMFEMSim):
            z_next, s, l = sim.step(z, s, l, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext)
        elif isinstance(sim, sk.sims.elastic.ElasticFEMSim):
            z_next = sim.step(z, z_dot,
                                Q_ext=BQB_ext, b_ext=Bb_ext)
            
        z_dot = (z_next - z) / sim.sim_params.h
        z = z_next.copy()

        Y = sim.X + (sim.B @ z).reshape(-1, sim.dim)
        mesh.update_vertex_positions(Y)
    
    
    ps.set_user_callback(callback)
    ps.show()
    
    

if __name__ == "__main__":  
    dirname =  os.path.dirname(__file__)


    configs = [crabConfig()]
    for c in configs:
        print(c.name)
        [X, T] = load_mesh(c.geometry_path)
        dim = X.shape[1]
        X = normalize_mesh(X)
        
        c.max_iter = 1
        W, E, B, cI, cW, labels = compute_subspace(X, T, c.m, c.k, 
                                                   mu=c.ym, bI=c.bI)

        # video_path_mfem = dirname + "/results/drop2/" + c.name + "_mfem.mp4"
        mfem_sim = create_mfem_sim(X, T, c.ym, c.rho, 
                                c.h, c.max_iter, 
                                c.do_line_search,
                                B=B, cI=cI, cW=cW)
        
        interactive_click_demo(mfem_sim, c.bI)