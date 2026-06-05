"""Index permutations from MATLAB/C++ column-major layout to NumPy row-major.

Flat gradient and Hessian blocks produced by legacy C++/MATLAB code use a
different component ordering than the row-major ``F`` layout used throughout
simkit. The constants below reindex vectors and matrices when importing those
arrays.

Attributes
----------
_4vector_2D_ordering_, _9vector_3D_ordering_ : list of int
    Permutations for flattened 2D (4) and 3D (9) gradient vectors.
_4x4matrix_2D_ordering_, _9x9matrix_3D_ordering_ : list of int
    Permutations for flattened 2D (4×4) and 3D (9×9) Hessian blocks.
y : np.ndarray
    Precomputed flat index map for pairing two 9-vectors in 3D (used internally
    when building higher-order reindexing tables).
"""

_4vector_2D_ordering_ = [0, 2, 1, 3]

_9vector_3D_ordering_ = [0, 3, 6, 1, 4, 7, 2, 5, 8]

# ordering that takes us from matlab to python
_4x4matrix_2D_ordering_ = [0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15]

# ordering that takes us from matlab to python
_9x9matrix_3D_ordering_ = [0, 3, 6, 1, 4, 7, 2, 5, 8, 27, 30, 33, 28, 31, 34, 29, 32,
       35, 54, 57, 60, 55, 58, 61, 56, 59, 62, 9, 12, 15, 10, 13, 16, 11,
       14, 17, 36, 39, 42, 37, 40, 43, 38, 41, 44, 63, 66, 69, 64, 67, 70,
       65, 68, 71, 18, 21, 24, 19, 22, 25, 20, 23, 26, 45, 48, 51, 46, 49,
       52, 47, 50, 53, 72, 75, 78, 73, 76, 79, 74, 77, 80]

