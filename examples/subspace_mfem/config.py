
import sys
sys.path.append("../../")
import simkit as sk
import numpy as np


class beamConfig:
    def __init__(self):
        self.ym = 1e12
        self.h = 1e-2
        self.rho = 1e3
        self.m = 5
        self.k = 100
        self.max_iter = 3
        self.do_line_search = False
        self.name = "beam"
        self.geometry_path = sk.filesystem.get_data_directory() + "/2d/beam/beam.obj"
        self.bI = np.array([0, 785])
        
        
class beamConfig:
    def __init__(self):
        self.ym = 1e12
        self.h = 1e-2
        self.rho = 1e3
        self.m = 5
        self.k = 100
        self.max_iter = 3
        self.do_line_search = False
        self.name = "beam"
        self.geometry_path = sk.filesystem.get_data_directory() + "/2d/beam/beam.obj"
        self.bI = np.array([0, 785])
        
        
class TConfig:
    def __init__(self):
        self.ym = sk.filesystem.get_data_directory() + "/2d/T/mu.npy"
        self.h = 1e-2
        self.rho = 1e3
        self.m = 5
        self.k = 96
        self.max_iter = 3
        self.do_line_search = True
        self.name = "T"
        self.geometry_path = sk.filesystem.get_data_directory() + "/2d/T/T.obj"
        self.bI = sk.filesystem.get_data_directory() + "/2d/T/bI.npy"
        
        
class cthuluConfig:
    def __init__(self):
        self.ym = 1e12
        
        self.h = 1e-2
        self.rho = 1e3
        self.m = 5
        self.k = 100
        self.max_iter = 3
        self.do_line_search = False
        self.name = "cthulu"
        self.geometry_path = sk.filesystem.get_data_directory() + "/2d/cthulu/cthulu.obj"
        self.bI = np.array([0, 785])
        
        
class crabConfig:
    def __init__(self):
        self.ym = sk.filesystem.get_data_directory() + "/3d/crab/mu.npy"
        self.h = 1e-2
        self.rho = 1e3
        self.m = 16
        self.k = 400
        self.max_iter = 3
        self.do_line_search = True
        self.name = "crab"
        self.geometry_path = sk.filesystem.get_data_directory() + "/3d/crab/crab.mesh"
        self.bI = sk.filesystem.get_data_directory() + "/3d/crab/bI.npy"