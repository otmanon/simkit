import numpy as np

def joint_edge_map(edges, joints):
    """
    Parameters
    ----------
    edges : (M,2) array_like of int
        Each row is an edge as (v1, v2).
    joints : (N,3) array_like of int
        Each row is a joint as (v_left, v_center, v_right).

    Returns
    -------
    joint_edges : (N,2) ndarray of int
        For each joint, gives the indices into 'edges' of the two edges
        that make up the joint: [(v_left,v_center), (v_center,v_right)].
    """
    edges = np.asarray(edges, dtype=int)
    joints = np.asarray(joints, dtype=int)

    # Build a dictionary from sorted vertex pairs -> edge index
    edge_map = {tuple(sorted(e)): idx for idx, e in enumerate(edges)}

    joint_edges = np.zeros((len(joints), 2), dtype=int)

    for j_idx, (i, j, k) in enumerate(joints):
        e1 = edge_map[tuple(sorted((i, j)))]
        e2 = edge_map[tuple(sorted((j, k)))]
        joint_edges[j_idx, 0] = e1
        joint_edges[j_idx, 1] = e2  

    return joint_edges