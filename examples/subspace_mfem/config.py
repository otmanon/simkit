
import sys
sys.path.append("../../")
import simkit as sk
import numpy as np

# class Config:
#     def __init__(self):
#         self.ym = sk.filesystem.get_data_directory() + "/2d/T/mu.npy"
#         self.h = 1e-2
#         self.rho = 1e3
#         self.m = 5
#         self.k = 96
#         self.max_iter = 3
#         self.do_line_search = True
#         self.name = "T"
#         self.geometry_path = sk.filesystem.get_data_directory() + "/2d/T/T.obj"
#         self.bI = sk.filesystem.get_data_directory() + "/2d/T/bI.npy"
        
      
class TConfig():
    def __init__(self):
        self.ym = sk.filesystem.get_data_directory() + "/2d/T/mu.npy"
        self.h = 1e-2
        self.rho = 1e3
        self.m = None
        self.k = None
        self.max_iter = 3
        self.do_line_search = True
        self.name = "tshape"
        self.geometry_path = sk.filesystem.get_data_directory() + "/2d/T/T.obj"
        self.bI = sk.filesystem.get_data_directory() + "/2d/T/bI.npy"
        self.eye_pos = np.array([0, 0, 3])
        self.look_at = np.array([0, 0, 0])
        
        self.pullI = np.array([14])
        self.pull_disp = np.array([[-1, 0]])
        
        
class cthuluConfig():
    def __init__(self):
        self.ym = 1e12
        
        self.h = 1e-2
        self.rho = 1e3
        self.m = 5
        self.k = 100
        self.max_iter = 3
        self.do_line_search = True
        self.name = "cthulu"
        self.geometry_path = sk.filesystem.get_data_directory() + "/2d/cthulu/cthulu.obj"
        self.bI = np.array([0, 785])
        self.eye_pos = np.array([0, 0, 3])
        self.look_at = np.array([0, 0, 0])
        
        self.pullI = np.array([149])
        self.pull_disp = np.array([[-1, 0]])
        
class crabConfig():
    def __init__(self):
        self.ym = sk.filesystem.get_data_directory() + "/3d/crab/mu.npy"
        self.h = 1e-2
        self.rho = 1e3
        self.m = 12
        self.k = 300
        self.max_iter = 2
        self.do_line_search = True
        self.name = "crab"
        self.geometry_path = sk.filesystem.get_data_directory() + "/3d/crab/crab.mesh"
        self.bI = sk.filesystem.get_data_directory() + "/3d/crab/bI.npy"
        self.eye_pos = np.array([-0.5, -0.5, 3])
        self.look_at = np.array([-0.5, -0.5, 0])
        
        
class gatormanConfig():
    def __init__(self):
        self.ym = sk.filesystem.get_data_directory() + "/3d/gatorman/mu.npy"
        self.h = 1e-2
        self.rho = 1e3
        self.m = 10
        self.k = 300
        self.max_iter = 3
        self.do_line_search = True
        self.name = "gatorman"
        self.geometry_path = sk.filesystem.get_data_directory() + "/3d/gatorman/gatorman.mesh"
        self.bI = sk.filesystem.get_data_directory() + "/3d/gatorman/bI.npy"
        self.eye_pos = np.array([-2, -0, 2])
        self.look_at = np.array([0, 0, 0])
        
        self.pullI = sk.filesystem.get_data_directory() + "/3d/gatorman/pullI.npy"
        self.pull_disp = np.array([[0, 0, 2]])
        
        
        
        
        
        