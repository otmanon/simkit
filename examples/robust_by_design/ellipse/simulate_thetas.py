
import os

from main import *

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = current_dir + '/results/'
    os.makedirs(result_dir, exist_ok=True)

    X, T, Zs =  simulate_shape_with_perturbation_func(np.array([1.0, 0.5]), 0)
    view_animation(X, T, Zs - X.reshape(-1, 1), V_ground, E_ground)#, 
              #  path=os.path.join(result_dir, 'ellipse_falling_sim_0.mp4'))


    X, T, Zs = simulate_shape_with_perturbation_func(np.array([1.0, 0.5]), np.pi/4)
    view_animation(X, T, Zs - X.reshape(-1, 1), V_ground, E_ground)#, 
         #       path=os.path.join(result_dir, 'ellipse_falling_sim_45deg.mp4'))


    X, T, Zs =  simulate_shape_with_perturbation_func(np.array([1.0, 0.5]), np.pi/2)
    view_animation(X, T, Zs - X.reshape(-1, 1), V_ground, E_ground)#,
            #    path=os.path.join(result_dir, 'ellipse_falling_sim_90deg.mp4'))

        