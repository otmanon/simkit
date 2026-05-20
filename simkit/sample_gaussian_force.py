import numpy as np



def sample_gaussian_force(Sigma_sqrt, mu, num_samples=1):
    
    # assert Sigma_sqrt.shape[1] == mu.shape[0], "Sigma_sqrt and mu must have the same number of rows"
    
    
    dim = Sigma_sqrt.shape[1]
    n = num_samples
    samples = np.random.randn(dim, n)
    
    return Sigma_sqrt @ samples + mu