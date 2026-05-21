import numpy as np

def area_normal_element(x0, x1, x2):
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x1 - x0
    
    n = np.cross(e1, -e2)
    return n

def area_normal_gradient_element(x0, x1, x2):
    
    def skew(x):
        '''
        Given n x 3 matrix, return n x 3 x 3 matrix of skew symmetric matrices.
        '''
        z = np.zeros((x.shape[0]))
        skew_matrix = np.array([[z, -x[:, 2], x[:, 1]],
                                [x[:, 2], z, -x[:, 0]],
                                [-x[:, 1], x[:, 0], z]])
        return skew_matrix.transpose(2, 1, 0)

    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x1 - x0
    
    dn_dx0 = skew(e0)
    dn_dx1 = skew(e1)
    dn_dx2 = skew(e2)
 
    dn_dx = np.concatenate([dn_dx0, dn_dx1, dn_dx2], axis=2)
    
    return dn_dx

def area_normal_hessian_element(x0, x1, x2):
    # This hessian is constant. I got this from finite differences, but can derive it. 
    H = np.array([[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.],
                    [ 0.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., -1.],
                    [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.],
                    [ 0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.]],  # dn_dx0

                [[ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., -1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.],
                    [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 1.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [-1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]],  # dn_dx1

                [[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
                    [-1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                    [ 1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]])  # dn_dx2
    
    num_elem = x0.shape[0]
    H = np.tile(H, (num_elem, 1, 1, 1))
    return H