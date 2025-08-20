
_4vector_2D_ordering_ = [0, 2, 1, 3]

_9vector_3D_ordering_ = [0, 3, 6, 1, 4, 7, 2, 5, 8]

# ordering that takes us from matlab to python
_4x4matrix_2D_ordering_ = [0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15]

#ordering that takes us from matlab to python
_9x9matrix_3D_ordering_ = [ 0,  3,  6,  1,  4,  7,  2,  5,  8, 27, 30, 33, 28, 31, 34, 29, 32,
       35, 54, 57, 60, 55, 58, 61, 56, 59, 62,  9, 12, 15, 10, 13, 16, 11,
       14, 17, 36, 39, 42, 37, 40, 43, 38, 41, 44, 63, 66, 69, 64, 67, 70,
       65, 68, 71, 18, 21, 24, 19, 22, 25, 20, 23, 26, 45, 48, 51, 46, 49,
       52, 47, 50, 53, 72, 75, 78, 73, 76, 79, 74, 77, 80]




import numpy as np


dim = 3
I = np.repeat(np.arange(9)[:, None], dim*dim , axis=1)
J = np.repeat(np.arange(9)[None, :], dim*dim , axis=0)


mI = np.array(_9vector_3D_ordering_)[I]
mJ = np.array(_9vector_3D_ordering_)[J]

y = (mI + dim*dim * mJ).flatten()