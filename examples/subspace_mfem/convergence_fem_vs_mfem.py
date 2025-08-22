import numpy as np
import simkit as sk

from config import *
from utils import *
from drop_fem_vs_mfem import simulate_drop_mfem, simulate_drop_fem, view_animation



if __name__ == "__main__":
    dirname =  os.path.dirname(__file__)

    configs = [cthuluConfig()]
    num_timesteps = 1
    for c in configs:
        print(c.name)
        
        c.rho = 1e0
        c.ym = 1e8
        # c.m = None
        # c.k = None
        c.max_iter = 100
        c.do_line_search = True
        
        
        c.bI = np.array([0])
    
        [X, T] = load_mesh(c.geometry_path)
        
        dim = X.shape[1]
        X = normalize_mesh(X)

        W, E, B, cI, cW, labels = compute_subspace(X, T, c.m, c.k, mu=c.ym)

        video_path_mfem = dirname + "/results/drop/" + c.name + "_mfem.mp4"
        mfem_sim = create_mfem_sim(X, T, c.ym, c.rho, 
                                c.h, c.max_iter, 
                                c.do_line_search,
                                B=B, cI=cI, cW=cW)
        
        [Zs, As, l, info_history] = simulate_drop_mfem(mfem_sim, c.bI,
                                    num_timesteps, 
                                    return_info=True)
        
        
        import polyscope as ps
        ps.init()
        ps.register_surface_mesh("mesh", X + (mfem_sim.B @ Zs[:, -1]).reshape(-1, dim), T)
        ps.register_surface_mesh("mesh2", X , T)
        ps.show()
        
        
        # view_animation(X, T, (mfem_sim.B @ Zs), 
        #     eye_pos=c.eye_pos,
        #     look_at=c.look_at)
        # ps.remove_all_structures()
        
        dp_hist = np.array(info_history[0]['dx'])[:, :, 0].T
        g_hist = mfem_sim.z_a_l_from_p(np.array(info_history[0]['g'])[:, :, 0].T)[0]
        
        step_sizes = np.array([info['alphas'] for info in info_history]).flatten()
        dx_hist = mfem_sim.z_a_l_from_p(dp_hist)[0]
        
        du = np.cumsum(dx_hist * step_sizes[None, :], axis=0)
        
        # import polyscope as ps
        # ps.init()
        # ps.reset_camera_to_home_view()
        # mesh0 = ps.register_surface_mesh("mesh0", X , T)
        # mesh1 = ps.register_surface_mesh("mesh1", X + (mfem_sim.B @ Zs[:, -1]).reshape(-1, dim), T)
        # ps.show()
        
        # for i in range(g_hist.shape[1]):
            
        #     mesh0.add_vector_quantity("g", g_hist[:,i].reshape(-1, dim),  enabled=True, defined_on='vertices')
            
        #     ps.frame_tick()
            
        #     mesh0.add_vector_quantity("dp", du[:,i].reshape(-1, dim),  enabled=True, defined_on='vertices',vectortype='ambient')
            
        #     ps.frame_tick()
            
        #     mesh0 = ps.register_surface_mesh("mesh0", X + (mfem_sim.B @ du[:,i]).reshape(-1, dim), T)
            
        #     ps.frame_tick()
            

    
        # du = np.cumsum(search_directions * step_sizes[:, None], axis=0)
        # ps.init()
        # ps.reset_camera_to_home_view()
        # mesh0 = ps.register_surface_mesh("mesh0", X , T)
        # mesh1 = ps.register_surface_mesh("mesh1", X + (mfem_sim.B @ Zs[:, -1]).reshape(-1, dim), T)
        # mesh2 = ps.register_surface_mesh("mesh2", X + (mfem_sim.B @ dx_hist[:, -2]).reshape(-1, dim), T)
        # ps.show()




        view_animation(X, T, (mfem_sim.B @ du))
        # view_animation(X, T, (mfem_sim.B @ Zs), 
        #             path=video_path_mfem, eye_pos=c.eye_pos,
        #             look_at=c.look_at)



        ############## FEM ############
        # video_path_fem = dirname + "/results/drop/" + c.name + "_fem.mp4"
        # fem_sim = create_fem_sim(X, T, c.ym, 
        #                         c.rho, c.h, 
        #                         c.max_iter,
        #                         c.do_line_search,
        #                         B=B, cI=cI, cW=cW)
        
        # [Zs, info_history] = simulate_drop_fem(fem_sim, c.bI, num_timesteps,
        #                     return_info=True)
        
        # import polyscope as ps
        # ps.init()
        # ps.register_surface_mesh("mesh", X + (fem_sim.B @ Zs[:, -1]).reshape(-1, dim), T)
        # ps.register_surface_mesh("mesh2", X , T)
        # ps.show()
        
        # dp_hist = np.array(info_history[0]['dx'])[:, : , 0]
        # search_directions = np.array(dp_hist)
        # step_sizes = np.array(info_history[0]['alphas'])
        # du = np.cumsum(search_directions * step_sizes[:, None], axis=0)
        # view_animation(X, T, (fem_sim.B @ du.T))
          
        # view_animation(X, T, (fem_sim.B @ Zs),
        #             path=video_path_fem, eye_pos=c.eye_pos,
        #             look_at=c.look_at)

