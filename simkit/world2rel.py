import numpy as np
def world2rel(P, P0):
    frames = P.shape[0]
    num_bones = P0.shape[0]

    Prel = np.zeros(P.shape)
    for frame in range(frames):
        for b in range(num_bones):
            # stack a row of [0 0 0 1] tp T0
            T0 = np.vstack((P0[b, :, :], np.array([0, 0, 0, 1])))

            # same with T
            T = np.vstack((P[frame,b, :, :], np.array([0, 0, 0, 1])))
            Trel = T @ np.linalg.inv(T0)
            Prel[frame, b, :, :] = Trel[:3, :]

    return Prel

