import numpy as np
import os
from pathlib import Path

from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir
from simkit.sims import *
from simkit.sims.pinned_pendulum.PinnedPendulumMFEMSim import *
from simkit.sims.pinned_pendulum.PinnedPendulumFEMSim import *
import matplotlib.pyplot as plt
from simkit.matplotlib.PointCloud import PointCloud
from simkit.matplotlib.Curve import Curve
from simkit.matplotlib.colors import *
dir = os.path.dirname(__file__)


class Experiment:

    def __init__():
        pass

    def render(self, X, path=None):
        
        if path is not None:
            dir = os.path.join(os.path.dirname(path), Path(path).stem) 
            os.makedirs(dir, exist_ok=True)

            
        x0 = np.vstack([[0, 0], X[:, 0].T])
        E = np.array([[0, 1]])
        
        y0 = np.array([[0, 0], [0, -1]])

        ref = Curve(y0, E, color=light_red, linewidth=4)
        pc_ref = PointCloud(y0, size=200, color=light_red, edgecolor='black', linewidth=0.1)
        
        cu = Curve(x0,E, color=light_blue, linewidth=4)
        pc = PointCloud(x0, size=200, color=light_blue, edgecolor='black', linewidth=0.1)


        for i in range(X.shape[1]):
            plt.title('Iteration : ' + str(i))
            xi = np.vstack([[0, 0], X[:, i].T])

            pc.update_vertex_positions(xi)
            cu.update_vertex_positions(xi)
            plt.pause(0.001)

            if path is not None:
                plt.savefig(dir + "/" + str(i).zfill(4) + ".png")

        for j in range(50):
            plt.title('Iteration : ' + str(i))
            xi = np.vstack([[0, 0], X[:, i].T])

            pc.update_vertex_positions(xi)
            cu.update_vertex_positions(xi)
            plt.pause(0.001)

            if path is not None:
                plt.savefig(dir + "/" + str(i+j).zfill(4) + ".png")

        if path is not None:
            video_from_image_dir(dir, path, fps=30)
            path_gif = path.replace(".mp4", ".gif")
            mp4_to_gif(path, path_gif)
        
        return


class FEMExperiment(Experiment):
    def __init__(self, mu=1):
        fem_sim_params = PinnedPendulumFEMSimParams(mu=mu)
        fem_sim_params.solver_p.max_iter = 1
        fem_sim = PinnedPendulumFEMSim(fem_sim_params)
        self.sim = fem_sim
    
    def simulate(self, num_timesteps=500):
        x = self.sim.rest_state()
        Xs = np.zeros((2, num_timesteps + 1))

        Xs[:, 0] = x.flatten()
        for i in range(num_timesteps):
            x = self.sim.step(x)
            Xs[:, i+1] = x.flatten()
            if np.linalg.norm(x - self.sim.p.y) < 1e-6:
                break
        Xs = Xs[:, :i+2]
        return Xs
    
class MFEMExperiment(Experiment):
    def __init__(self, mu=1):
        mfem_sim_params = PinnedPendulumMFEMSimParams(mu=mu)
        mfem_sim_params.solver_p.max_iter = 1
        mfem_sim = PinnedPendulumMFEMSim(mfem_sim_params)
        self.sim = mfem_sim

    def simulate(self, num_timesteps=500):
        [x, s, l] = self.sim.rest_state()
        Xs = np.zeros((2, num_timesteps +1))
        Xs[:, 0] = x.flatten()
        for i in range(num_timesteps):
            [x, s, l] = self.sim.step(x, s, l)

            Xs[:, i+1] = x.flatten()
            if np.linalg.norm(x - self.sim.p.y) < 1e-6:
                break
        Xs = Xs[:, :i+2]
        return Xs
    

results_dir = os.path.join(dir, 'results')

mus = [1e2, 1e4, 1e8]

plt.ion()
plt.figure()
plt.axis('equal')
plt.xlim([-0.5, 1.5])
plt.ylim([-1.5, 0.5])
plt.axis('off')


x0 = np.vstack([[0, 0], [1, 0.]])
E = np.array([[0, 1]])

y0 = np.array([[0, 0], [0, -1]])

# cu = Curve(x0, E, color=light_blue, linewidth=4)
# pc = PointCloud(x0, size=200, color=light_blue, edgecolor='black', linewidth=0.1)
# pc = PointCloud(x0[[0], :], size=200, color=black, edgecolor='black', linewidth=0.1)

# plt.show()
# plt.savefig(dir + '/fem_pendulum_setup.png', dpi=300)
for mu in mus:
    exp = FEMExperiment(mu=mu)
    X = exp.simulate()
    exp.render(X, path=dir + "/fem_pendulum_mu_" + str(mu) + ".mp4")


    # exp_mfem = MFEMExperiment(mu=mu)
    # X_mfem = exp_mfem.simulate()

    # exp_mfem.render(X_mfem, path=dir + "/mfem_pendulum_mu_" + str(mu) + ".mp4")
