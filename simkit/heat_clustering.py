from scipy.spatial import cKDTree
import numpy as np
import igl
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import simkit as sk

def distance_all_pairs(A, B):
    # Compute all pairwise distances between two sets of points
    tree = cKDTree(A)
    [D, I] = tree.query(B)
    return D, I

def dist_func_elem(I, data, T):
    """Distance function for elements."""
    return np.mean(sk.heat_distance_solve(data, T[I])[T], axis=1)

def dist_func_nodal(I, data):
    """Distance function for nodes."""
    return sk.heat_distance_solve(data, I)

def centroid(X, WX, labels, cluster_volumes, exclusion):
    """
    Compute the closest representative (approximate centroid) within each cluster.

    Parameters:
    - X: All point positions (n x d)
    - WX: Weighted positions (element-wise volume or mass times X)
    - labels: Cluster assignment for each point (length n)
    - cluster_volumes: Total volume per cluster (length k)
    - exclusion: Indices to exclude from centroid selection

    Returns:
    - XI: Indices of representative centroids (length k)
    """
    k = len(cluster_volumes)
    dim = X.shape[1]
    CX = np.zeros((k, dim))
    for d in range(dim):
        CX[:, d] = np.bincount(labels, WX[:, d], minlength=k)
    CX /= cluster_volumes[:, None]

    # Compute distance from centroid to closest point within the cluster
    XI = np.zeros(k, dtype=int)
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_indices = np.setdiff1d(cluster_indices, exclusion)
        _, idx = distance_all_pairs(X[cluster_indices], CX[i])
        XI[i] = cluster_indices[idx]
    return XI

def approximate_mediod(labels, cluster_volumes, dist_func, exclusion, num_samples=100):
    """
    Approximate the medoid for each cluster using a subset of candidates.

    Parameters:
    - labels: Cluster assignment for each element (length n)
    - cluster_volumes: Total volume of each cluster
    - dist_func: Function that computes distance from a source index to all other points
    - exclusion: Indices to exclude
    - num_samples: Number of candidates to sample per cluster

    Returns:
    - XI: Indices of approximate medoids (length k)
    """
    k = len(cluster_volumes)
    XI = np.zeros(k, dtype=int)

    for i in range(k):
        cluster_id = i
        cluster_indices = np.where(labels == cluster_id)[0]  # elements within the cluster
        cluster_indices = np.setdiff1d(cluster_indices, exclusion)

        num_samples = min(num_samples, len(cluster_indices))
        sample_indices = np.random.choice(cluster_indices, num_samples, replace=False)

        min_sum = np.inf
        current_idx = -1
        for idx in sample_indices:
            dist = dist_func(idx)[cluster_indices]
            sum_dist = np.sum(dist)
            if sum_dist < min_sum:
                min_sum = sum_dist
                current_idx = idx
        XI[i] = current_idx
    return XI

class HeatClusteringData:
    """
    Container for heat clustering precomputation.

    Attributes:
    - X: Points to cluster (either vertices or barycenters)
    - XI: Indices of initial cluster representatives (length k)
    - C: Current cluster centroid positions
    - vol: Volume or weight associated with each point
    - distFunc: Distance function used to compute heat distances
    - centroidFunc: Function to recompute cluster representatives
    """
    def __init__(self, X, XI, C, vol, distFunc, centroidFunc):
        self.X = X
        self.XI = XI
        self.C = C
        self.vol = vol
        self.distFunc = distFunc
        self.centroidFunc = centroidFunc

def heat_clustering_precomp(V, T, k=30, mu=None, mode='elem', exclusion=None, centroid_type="centroid"):
    """
    Precompute clustering data using heat geodesic distances.

    Parameters:
    - V: Vertex positions (n x 3)
    - T: Connectivity (triangle or tetrahedron indices)
    - k: Number of clusters
    - mu: Stiffness or weight per element (defaults to uniform)
    - mode: 'elem' for element-wise clustering, 'nodal' for vertex-wise
    - exclusion: Indices to exclude from seed/centroid selection
    - centroid_type: Either 'centroid' (weighted mean) or 'mediod' (minimum sum distance)

    Returns:
    - HeatClusteringData instance
    """
    if mu is None:
        mu = np.ones(len(T))

    # mass-lumping onto the diagonal
    if mode == 'elem':
        if T.shape[1] == 3:
            vol = 0.5 * igl.doublearea(V, T)
        else:
            vol = igl.volume(V, T)
        X = igl.barycenter(V, T)
    elif mode == 'nodal':
        vol = igl.massmatrix(V, T)
        vol = vol.sum(1).reshape(-1, 1)
        vol = np.array(vol).flatten()
        X = V
    else:
        raise ValueError('Unknown mode: {}'.format(mode))

    dim = X.shape[1]
    n = len(X)
    XI = np.zeros(k, dtype=int)
    C = np.zeros((k, dim))
    rng = np.random.default_rng(seed=0)

    # Exclusion point handling
    probabilities = vol.copy()
    if exclusion is not None:
        probabilities[exclusion] = 0

    probabilities /= np.sum(probabilities)

    # Initialize first seed
    idx = rng.choice(n, 1, p=probabilities.flatten())
    XI[0] = idx
    C[0, :] = X[idx]

    # Precompute heat data
    t = igl.avg_edge_length(V, T)**2 / np.min(mu)
    L = sk.dirichlet_laplacian(V, T, mu=mu)
    heat_data = sk.heat_distance_precompute(V, T, L=-L, t=t)

    # Define distance function
    if mode == 'elem':
        distFunc = partial(dist_func_elem, data=heat_data, T=T)
    else:
        distFunc = partial(dist_func_nodal, data=heat_data)

    # Initialize minimum distances
    minDist = np.full((n,1), np.inf)

    # kmeans++ initialization
    for ii in range(1, k):
        minDist = np.minimum(minDist, distFunc(XI[ii - 1]))
        sampleProbability = np.maximum(minDist, 1e-12)
        sampleProbability[exclusion] = 1e-12
        sampleProbability /= np.sum(sampleProbability)

        if not np.isfinite(sampleProbability).all():
            raise ValueError('Sample probability is not finite')

        XI[ii] = rng.choice(n, 1, p=sampleProbability.flatten())
        C[ii, :] = X[XI[ii]]

    # Choose centroid function
    if centroid_type == "centroid":
        WX = vol[:, None] * X # weighted X for centroids
        centroidFunc = lambda labels, cluster_volumes : centroid(X, WX, labels, cluster_volumes, exclusion)
    elif centroid_type == "mediod":
        centroidFunc = lambda labels, cluster_volumes : approximate_mediod(
            labels, cluster_volumes, distFunc, exclusion, num_samples=50)
    else:
        raise ValueError(f'Unknown centroid type: {centroid_type}')

    return HeatClusteringData(X, XI, C, vol, distFunc, centroidFunc)

def heat_clustering_solve(data, max_iters=50, save_intermediate=True, tolerance=1e-5):

    """
    Perform Lloyd relaxation clustering using heat geodesic distances.

    Parameters:
    - data: HeatClusteringData object
    - max_iters: Maximum number of k-means-like refinement steps
    - save_intermediate: Whether to store and return history of assignments
    - tolerance: Convergence threshold (relative centroid movement)

    Returns:
    - labels: Final cluster assignments
    - C: Final centroid positions
    - dist: Distance matrix (|X| x k)
    - clusters: List of cluster assignments at each iteration
    - centroids: List of centroids at each iteration
    """
    X, XI, C, vol, distFunc = data.X, data.XI, data.C, data.vol, data.distFunc
    k = len(C)
    dist = np.zeros((len(X), k))
    L = np.linalg.norm(np.max(X, axis=0) - np.min(X, axis=0))

    clusters=[]
    centroids=[]

    # compute initial partitions and distances
    # This is usually just 2x faster using threading
    with ThreadPoolExecutor() as executor:
        dist[:, :] = np.array(list(executor.map(lambda j: distFunc(XI[j]), range(k)))).T
    labels = np.argmin(dist, axis=1)

    if save_intermediate:
        clusters.append(labels)
        centroids.append(C)

    for i in range(max_iters):
        C0 = C.copy()

        cluster_volumes = np.bincount(labels, vol, minlength=k)
        XI = data.centroidFunc(labels, cluster_volumes)
        C = X[XI]

        conv = np.linalg.norm(C - C0) / (L * k)
        minvol = np.min(cluster_volumes) / np.sum(cluster_volumes)
        maxvol = np.max(cluster_volumes) / np.sum(cluster_volumes)

        with ThreadPoolExecutor() as executor:
            dist[:, :] = np.array(list(executor.map(lambda j: distFunc(XI[j]), range(k)))).T
        labels = np.argmin(dist, axis=1)
        if save_intermediate:
            clusters.append(labels)
            centroids.append(C)

        print(f'Iteration {i}, Conv: {conv:.4g}, Min/Max vol ratio: {minvol:.4g}, {maxvol:.4g}')

        if conv < tolerance:
            break
    return labels, C, dist, clusters, centroids