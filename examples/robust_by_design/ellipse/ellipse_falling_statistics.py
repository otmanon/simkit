import os
from cv2 import ellipse
import igl
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import simkit as sk

from examples.robust_by_design.ellipse.main import *


current_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = current_dir + '/results/'
os.makedirs(result_dir, exist_ok=True)


p = np.array([1.0, 0.5])
thetas = np.linspace(0, np.pi/2, 3)



com_ys_all = evaluate_com_height_across_thetas(thetas, p)

coms_vars_plots(com_ys_all, thetas, result_dir, legend=True)