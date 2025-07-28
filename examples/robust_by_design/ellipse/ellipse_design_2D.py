import os
import igl
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from examples.modal_muscles.animation_viewers import animation_viewer_2D
import simkit as sk

import polyscope as ps

def view_animation(X, T, Zs, V_ground, E_ground):
    
    dim = X.shape[1]
    
    ps.init()
    ps.set_ground_plane_mode("none")

    ground_mesh = ps.register_curve_network("ground", V_ground, E_ground)
    surface_mesh = ps.register_surface_mesh("circle", X, T, edge_width=1.0)
    ps.look_at(np.array([0.0, height/2, 5.0]), np.array([0.0, height/2, 0.0]))
    # ps.reset_camera_to_home_view()
    for i in range(Zs.shape[1]):
        surface_mesh.update_vertex_positions(X + Zs[:, [i]].reshape(-1, dim))
        ps.frame_tick()
    
    

def simulate(sim):
    z, z_dot = sim.rest_state()

    Zs = np.zeros((z.shape[0], num_timesteps))
    for i in range(num_timesteps):
        z_next = sim.step(z, z_dot)
        z_dot = (z_next - z) /sim.params.h
        z = z_next.copy()
        Zs[:, [i]] = z
        
    return Zs
height = 4.0
dim = 2

num_timesteps = 500
k_contact = 1e6
contact_p = np.array([0.0, 0.0])
contact_n = np.array([0.0, 1.0])

# get two endpoints  on curve with normal n that goes through contact_p
tangent = np.array([1.0, 0.0])
ground_plane_width = 2
tangent[0] = contact_n[1]
tangent[1] = -contact_n[0]
tangent = tangent / np.linalg.norm(tangent)
contact_p1 = contact_p + tangent * ground_plane_width / 2
contact_p2 = contact_p - tangent * ground_plane_width / 2
V_ground = np.concatenate([[contact_p1], [contact_p2]], axis=0)
E_ground = np.arange(V_ground.shape[0]).reshape(-1, 2)
rho = 1e3
ym = 1e5
ag = -9.8


num_samples = 10
thetas = np.linspace(0, np.pi/2, num_samples)   
def func(p):
    [Ve, Ee] = sk.ellipse_outline(a=p[0], b=p[1], n=25)
    [X0, T, _, _, _] = igl.triangle.triangulate(Ve, Ee, flags='qa0.02')

    M = sk.massmatrix(X0, T)
    Me = sp.sparse.kron(M, sp.sparse.identity(dim))
    Se = sp.sparse.kron(np.ones((1, X0.shape[0])), sp.sparse.identity(dim))
    SM = Se @ Me
    m = M.diagonal()
    total_mass = np.sum(m)
    x_com_ys = np.zeros((num_samples, 1))
    
    for i in range(num_samples):
        theta = thetas[i]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        X = X0 @ R
        X += np.array([0.0, height])

        g = sk.gravity_force(X, T, a=ag, rho=rho)
        sim_params = sk.sims.elastic.ElasticFEMSimParams(rho=rho, ym=ym, b0 = -g)
        sim = sk.sims.elastic.ElasticFEMSim(X, T, params=sim_params)
        def new_energy(z):
            e_contact = sk.energies.contact_springs_plane_energy(z.reshape(-1, dim), k_contact, contact_p, contact_n)
            e = sim.energy(z) + e_contact
            return e
        def new_energy_gradient(z):
            g_contact = sk.energies.contact_springs_plane_gradient(z.reshape(-1, dim), k_contact, contact_p, contact_n)
            g = sim.gradient(z) + g_contact
            return g
        def new_energy_hessian(z):
            H_contact = sk.energies.contact_springs_plane_hessian(z.reshape(-1, dim), k_contact, contact_p, contact_n)
            H = sim.hessian(z) + H_contact
            return H
        sim.solver.energy_func = new_energy
        sim.solver.gradient_func = new_energy_gradient
        sim.solver.hessian_func = new_energy_hessian

        Zs = simulate(sim)
        # view_animation(X, T, Zs - X.reshape(-1, 1), V_ground, E_ground)

        com = (SM @ (Zs)) / total_mass
        com_max_y = np.max(com[1, :])
        
        x_com_y = com_max_y
        x_com_ys[i] = x_com_y
        
    x_com_ys_mean = np.mean(x_com_ys)
    x_com_ys_std = np.std(x_com_ys)
    print("STD(X_COM_Y) : ", x_com_ys_std)
    return x_com_ys_std

y0 = np.array([1.0, 0.5])
y_next = sp.optimize.minimize(func, y0)

print(y_next)

# view_animation(X, T, Zs - X.reshape(-1, 1), V_ground, E_ground)
    
    
    
    

# import polyscope as ps
# ps.init()
# ps.register_curve_network("circle", Ve, Ee)
# ps.register_surface_mesh("circle", V, F, edge_width=1.0)

# ps.show()

# set up ellipse design space



# set up objective 
