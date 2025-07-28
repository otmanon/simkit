import os
import igl
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from examples.modal_muscles.animation_viewers import animation_viewer_2D
import simkit as sk
from pathlib import Path
import polyscope as ps


height = 2
dim = 2
num_timesteps = 250
k_contact = 1e6
k_dyn_friction = 1e6
contact_p = np.array([0.0, 0.0])
contact_n = np.array([0.0, 1.0])
contact_n = contact_n / np.linalg.norm(contact_n)
tangent = np.array([contact_n[1], -contact_n[0]])
ground_plane_width = 100
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
pr= 0.45

current_dir = os.path.dirname(__file__)

        
X0, _, _, T, _, _ = igl.readOBJ(current_dir + "/circle_25.obj")
X0 = X0[:, [0, 1]]

def evaluate_com_height_across_thetas(thetas, p, return_Zs = False):
    com_ys_all = np.zeros((thetas.shape[0], num_timesteps))
    if return_Zs:
        Zs_list = []   
        X_list = [] 
    for i, theta in enumerate(thetas):
        X,  Zs = simulate_shape_with_perturbation_func(p, theta)
        M = sk.massmatrix(X, T)
        Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        Se = sp.sparse.kron(np.ones((1, X.shape[0])), sp.sparse.identity(dim))
        SM = Se @ Me
        m = M.diagonal().sum()
        com_ys = (SM @ Zs)[1, :]/m
        com_ys_all[i, :] = com_ys
    
        if return_Zs:
            Zs_list.append(Zs)
            X_list.append(X)
    if return_Zs:
        return com_ys_all, Zs_list, X_list
    else:
        return com_ys_all

def ellipse_falling_sim(X, T, ag, rho, ym, pr):
        
    g = sk.gravity_force(X, T, a=ag, rho=rho)
    sim_params = sk.sims.elastic.ElasticFEMSimParams(rho=rho, ym=ym, pr=pr, b0 = -g, material='neo-hookean')
    sim = sk.sims.elastic.ElasticFEMSim(X, T, params=sim_params)
    def new_energy(z):
        e_contact, contacting_inds = sk.energies.contact_springs_plane_energy(z.reshape(-1, dim), k_contact, contact_p, contact_n,
                                                             return_contact_inds=True)
        
        if contacting_inds is not None:
            tangents = np.repeat(tangent[None, :], contacting_inds.shape[0], axis=0)
            
            S = sk.selection_matrix(contacting_inds.flatten(), X.shape[0])
            e_dyn_friction = sk.energies.quadratic_dynamic_friction_energy(z, sim.z_curr,
                                                                           tangents, S,
                                                                          k_dyn_friction)
        else:
            e_dyn_friction = 0.0
            
        e = sim.energy(z) + e_contact + e_dyn_friction
        return e
    def new_energy_gradient(z):
        g_contact, contacting_inds = sk.energies.contact_springs_plane_gradient(z.reshape(-1, dim), k_contact, contact_p, contact_n,
                                                                                return_contact_inds=True)
        if contacting_inds is not None:
            tangents = np.repeat(tangent[None, :], contacting_inds.shape[0], axis=0)
            S = sk.selection_matrix(contacting_inds.flatten(), X.shape[0])
            g_dyn_friction = sk.energies.quadratic_dynamic_friction_gradient(z, sim.z_curr,
                                                                         tangents, S, k_dyn_friction)
        else:
            g_dyn_friction = np.zeros(z.shape)
        g = sim.gradient(z) + g_contact + g_dyn_friction
        return g
    def new_energy_hessian(z):
        H_contact, contacting_inds = sk.energies.contact_springs_plane_hessian(z.reshape(-1, dim), k_contact, contact_p, contact_n,
                                                                                return_contact_inds=True)
        if contacting_inds is not None:
            tangents = np.repeat(tangent[None, :], contacting_inds.shape[0], axis=0)
            S = sk.selection_matrix(contacting_inds.flatten(), X.shape[0])
            H_dyn_friction = sk.energies.quadratic_dynamic_friction_hessian(z, sim.z_curr,
                                                                         tangents, S, k_dyn_friction)
        else:
            H_dyn_friction = sp.sparse.csc_matrix((z.shape[0], z.shape[0]))
        H = sim.hessian(z) + H_contact + H_dyn_friction
        return H
    sim.solver.energy_func = new_energy
    sim.solver.gradient_func = new_energy_gradient
    sim.solver.hessian_func = new_energy_hessian
    
    return sim


def view_animation(X, T, Zs, V_ground, E_ground,
                   path=None,  fps=60, 
                    material='clay'):
    
    dim = X.shape[1]
    if path is not None:
        stem = Path(path).stem
        dir = Path(path).parent
        dirstem = os.path.join(dir, stem)
        os.makedirs(dirstem, exist_ok=True)

    ps.init()
    ps.set_ground_plane_mode("none")

    ground_mesh = ps.register_curve_network("ground", V_ground, E_ground, material=material)
    surface_mesh = ps.register_surface_mesh("circle", X, T, edge_width=1.0, material=material)
    ps.look_at(np.array([0.0, height/2, 5.0]), np.array([0.0, height/2, 0.0]))
    # ps.reset_camera_to_home_view()
    
    for i in range(Zs.shape[1]):
        surface_mesh.update_vertex_positions(X + Zs[:, [i]].reshape(-1, dim))
        ps.frame_tick()
        if path is not None:
            ps.screenshot(dirstem + "/" + str(i + 1).zfill(4) + ".png", transparent_bg=True)

    if path is not None:
        sk.filesystem.video_from_image_dir(dirstem, path, fps=fps)
        sk.filesystem.mp4_to_gif(path, path.replace(".mp4", ".gif"))

    ps.remove_all_structures()
def evaluate_shape(p):
    X = X0.copy()
    X[:, 0] *= p[0]
    X[:, 1] *= p[1]
    return X


# print first guess shape in polyscope
def save_shape(filepath, p):
    X = evaluate_shape(p)
    ps.init()
    E = igl.boundary_facets(T)[0]
    ps.register_curve_network('mesh', X, E, material="flat")
    ps.set_ground_plane_mode("none")
    ps.look_at([0, 0, 5], [0, 0, 0])
    ps.register_surface_mesh("mesh", X, T, material="flat", edge_width=1.0)
    ps.screenshot(filepath, True)
    ps.show()
    ps.remove_all_structures()
    return

def simulate_shape_with_perturbation_func(p, theta):
 
    X = evaluate_shape(p)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    X = X @ R
    X += np.array([0.0, height])

    sim = ellipse_falling_sim(X, T, ag, rho, ym, pr)
    Zs = simulate(sim)
    return X,Zs


    

def simulate(sim):
    z, z_dot = sim.rest_state()

    Zs = np.zeros((z.shape[0], num_timesteps))
    for i in range(num_timesteps):
        z_next = sim.step(z, z_dot)
        z_dot = (z_next - z) /sim.params.h
        z = z_next.copy()
        Zs[:, [i]] = z
        
    return Zs

        
        

# get two endpoints  on curve with normal n that goes through contact_p
def coms_vars_plots(com_ys_all, thetas, result_dir, legend=False):
    time = np.arange(com_ys_all.shape[1]) * 1e-2
    
    for i in range(thetas.shape[0]):
        plt.plot(time, com_ys_all[i, :], label='$\\theta = $ ' + str(np.round(thetas[i], 2)))
        
    plt.title('COM height through simulation')
    plt.xlabel('time (s)')
    plt.ylabel('COM height (m)')
    plt.ylim((0, height))
    if legend:
        plt.legend()
    plt.savefig(os.path.join(result_dir, 'com_y_plot_numthetas' + str(thetas.shape[0]) + '.png'), dpi=300)
    plt.show()
    plt.clf()
        
    # variance through time
    var = np.var(com_ys_all, axis=0)


    plt.plot(time, var)
        
    plt.title('Variance of COM height through simulation')
    plt.xlabel('time (s)')
    plt.ylabel('Variance of height (m^2)')
    plt.savefig(os.path.join(result_dir, 'var_com_y_plot_numthetas' + str(thetas.shape[0]) + '.png'), dpi=300)
    plt.show()
    



    # import polyscope as ps
    # ps.init()
    # ps.register_curve_network("circle", Ve, Ee)
    # ps.register_surface_mesh("circle", V, F, edge_width=1.0)

    # ps.show()

    # set up ellipse design space



    # set up objective 
