import numpy as np
import simkit as sk

from config import *
from utils import *




if __name__ == "__main__":
    dirname =  os.path.dirname(__file__)

    configs = [crabConfig()]
    
    
    # get young's modulus
    for c in configs:
        [X, T] = load_mesh(c.geometry_path)
        dim = X.shape[1]
        X = normalize_mesh(X)
        if isinstance(c.ym, str):
            ym = np.load(c.ym)
        else:
            ym = c.ym
            


        W, E, B, cI, cW, labels = compute_subspace(X, T, c.m, c.k, mu=c.ym, bI=c.bI)

        # visualize skinning eigenmodes
        
        # visualize cubature points
        
        # visualize stiffness

