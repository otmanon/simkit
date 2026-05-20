
import numpy as np
import scipy as sp
from itertools import combinations
def combinatorial_expansion(sigma_f_list, choose_number):

    # given a list of r covariance matrices, 

    # combine them in groups of choose_number by summing them up
    # return the expanded list of covariance matrices and the indices of the covariance matrices that were combined


    indices = np.arange(len(sigma_f_list))
    force_mixture_subspaces = np.array((list(combinations(indices, choose_number)))).tolist()

    sigma_f_list_expanded = np.array(list(combinations(sigma_f_list, choose_number)))

    sigma_f_list_expanded_sum = (np.sum(sigma_f_list_expanded, axis=1)/ choose_number).tolist()
    return sigma_f_list_expanded_sum, force_mixture_subspaces
        
def gmm_force_expansion(sigma_f_list, max_expansion=4):
        
    """
    Given a r force covariance matrices, 
    Create an expanded list of force covariance matrices whereby a user operates with combinations of the original force covariance matrices.
    The expanded list will contain all possible combinations of the original force covariance matrices, up to a limit of max_expansion
    """
    sigma_f_list_expanded = []
    force_mixture_subspaces = []
    r = len(sigma_f_list)

    for expansion in range(1, max_expansion+1):
        sigma_f_list_ei, force_mixture_subspace_i = combinatorial_expansion(sigma_f_list, expansion)    
        sigma_f_list_expanded.extend(sigma_f_list_ei)
        force_mixture_subspaces.extend(force_mixture_subspace_i)

    return sigma_f_list_expanded, force_mixture_subspaces