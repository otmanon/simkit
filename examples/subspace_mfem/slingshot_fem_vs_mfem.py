import numpy as np
import sys
sys.path.append("../../")

import igl
import polyscope as ps
import scipy as sp
import simkit as sk
import os

from utils import *
from config import *

dirname =  os.path.dirname(__file__)


configs = [ gatormanConfig(), TConfig()]
pull_timesteps = 100
free_timesteps = 300
for c in configs:
    print(c.name)
    [X, T] = load_mesh(c.geometry_path)
    dim = X.shape[1]
    X = normalize_mesh(X)


    W, E, B, cI, cW, labels = compute_subspace(X, T, c.m, c.k, mu=c.ym, bI=c.bI)

    # sk.polyscope.view_scalar_modes(X, T, W)
    
    video_path_mfem = dirname + "/results/slingshot/" + c.name + "_mfem.mp4"
    mfem_sim = create_mfem_sim(X, T, c.ym, c.rho, 
                            c.h, c.max_iter, 
                            c.do_line_search,
                            B=B, cI=cI, cW=cW)
    
    [Zs, As] = simulate_slingshot_mfem(mfem_sim, c.bI, 
                                       c.pullI,
                                       c.pull_disp,
                                       pull_timesteps,
                                       free_timesteps, 
                                       return_info=False)
    view_animation(X, T, (mfem_sim.B @ Zs), 
                path=video_path_mfem,
                eye_pos=c.eye_pos,
                look_at=c.look_at)


    video_path_fem = dirname + "/results/slingshot/" + c.name + "_fem.mp4"
    fem_sim = create_fem_sim(X, T, c.ym, 
                            c.rho, c.h, 
                            c.max_iter,
                            c.do_line_search,
                            B=B, cI=cI, cW=cW)
    Zs = simulate_slingshot_fem(fem_sim, c.bI, 
                                c.pullI,
                                c.pull_disp,
                                pull_timesteps,
                                free_timesteps, 
                                return_info=False)
    view_animation(X, T, (fem_sim.B @ Zs),
                path=video_path_fem,
                eye_pos=c.eye_pos,
                look_at=c.look_at)

