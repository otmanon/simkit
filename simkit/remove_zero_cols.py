
import numpy as np
import scipy as sp

def remove_zero_cols(B, thresh=1e-12):

    B_col_sum = np.array(np.sum(np.abs(B), axis=0))
    is_all_zeros = (B_col_sum < thresh).flatten()
    B_non_zeros = B[:, ~is_all_zeros]
    return B_non_zeros, is_all_zeros

