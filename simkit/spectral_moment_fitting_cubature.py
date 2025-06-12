import igl
import scipy as sp
import numpy as np
from simkit import spectral_clustering, volume, average_onto_simplex, spectral_clustering, pairwise_distance

from itertools import combinations_with_replacement

def compute_basis_integrals(volumes, B, p):
    """
    Compute the exact integrals of the polynomial basis over elements.

    Parameters:
    - volumes: Volumes of the elements (n x 1)
    - B: basis matrix (n x m)
    - p: Polynomial degree
    Returns:
    - basis_integrals: Precomputed basis integrals
    - A: Matrix of integrals for each basis function in each element
    """
    np.random.seed(0)

    # Generate polynomial combinations for the given degree p
    polynomial_combinations = []
    for i in range(p+1): # Include combinations of degree 0 to p
        polynomial_combinations.extend(combinations_with_replacement(range(B.shape[1]), i))

    # Compute products for all combinations
    products = np.ones((B.shape[0], len(polynomial_combinations)))
    for k, idx in enumerate(polynomial_combinations):
        products[:, k] = np.prod(B[:, idx], axis=1)

    # Integrate over each element
    A = products.T * volumes

    # Sum over all elements to get the basis integrals
    basis_integrals = np.sum(A, axis=1)
    return basis_integrals, A


def cubature_sampling(A, basis_integrals, num_to_sample, tolerance=1e-8, only_positive=True):
    """
    Perform cubature sampling to select points based on least-squares error minimization.

    Parameters:
    - A: Matrix of integrals for each basis function in each element (N x m)
    - basis_integrals: Precomputed integrals for each basis function (N,)
    - num_to_sample: Number of points to sample per iteration
    - tolerance: Error tolerance for stopping criterion (default 1e-8)
    - only_positive: If True, only return positive weights and corresponding indices (default True)

    Returns:
    - q_idx: Indices of the cubature points selected
    - w: Weights computed from the least-squares solution
    - errors: List of errors over the iterations
    """
    # Total number of elements (from the second dimension of A)
    m = A.shape[1]

    # Initialize the indices of cubature points
    q_idx = np.zeros(m + 1, dtype=int)  # Can hold up to m points

    i = 0
    errors = []

    # Per-element errors for sampling probabilities
    element_errors = np.ones(m)

    while i < m:
        # Set error of already-sampled elements to zero
        element_errors[q_idx[:i]] = 0

        # Normalize errors for sampling
        element_errors /= np.sum(element_errors)

        # Sample new cubature points proportional to error
        # if m exceeds m - i, sample m - i points
        num_to_sample = min(num_to_sample, m - i - 1)
        if num_to_sample == 0:
            break

        samples = np.random.choice(m, num_to_sample, replace=False, p=element_errors)
        q_idx[i + 1:i + 1 + num_to_sample] = samples
        i += num_to_sample

        # Build the linear system df (rows = basis functions at sample points)
        df = np.zeros((basis_integrals.shape[0], i + 1))
        for j in range(i + 1):
            idx = q_idx[j]
            df[:, j] = A[:, idx]

        # Solve least-squares system for weights (non-negative)
        wn = sp.optimize.nnls(df, basis_integrals, maxiter=100000)[0]

        # Compute moment fitting error
        fitted_moments = df @ wn
        error = basis_integrals - fitted_moments

        element_errors = (A.T @ error)**2

        lsq_error = np.linalg.norm(error)
        relative_error = lsq_error / np.linalg.norm(basis_integrals)
        # print(f"# points: {i + 1}, # constraints: {len(basis_integrals)}, Absolute error: {lsq_error}, Relative error: {relative_error}")
        errors.append(relative_error)

        # Check if the error is below the tolerance
        if relative_error < tolerance:
            break

    # Filter positive weights and corresponding indices if `only_positive` is True
    if only_positive:
        pos_ids = np.where(wn > 1e-10)[0]
        wn_pos = wn[pos_ids]
        q_pos = q_idx[pos_ids]
        return q_pos, wn_pos, errors

    # Return all weights and indices if `only_positive` is False
    return q_idx, wn, errors

def compute_cluster_cubature(labels, V, T, W, p=2, num_to_sample=500, tolerance=1e-8):
    """
    Perform cubature sampling over a set of labeled clusters.

    For each cluster, a reduced cubature rule is computed to approximate integrals
    over that region using a subset of elements and associated weights.

    Parameters:
    - labels: Integer array of length |T| assigning each tetrahedron to a cluster (values in [0, k-1])
    - V: Vertex positions of the full mesh (|V| x 3)
    - T: Tetrahedral element connectivity (|T| x 4)
    - W: Per-element mode evaluations (|T| x num_modes)
    - p: Degree of polynomial combinations to integrate against (default 2)
    - num_to_sample: Maximum number of elements to sample per cluster (default 500)
    - tolerance: Error tolerance for cubature fitting (default 1e-8)

    Returns:
    - cubature_indices: Concatenated list of selected element indices across all clusters
    - cubature_weights: Corresponding cubature weights (volume-scaled) for the selected indices
    """
    num_labels = np.max(labels) + 1
    print(f"Number of clusters: {num_labels}")

    cubature_indices = []
    cubature_weights = []

    volumes = volume(V, T).flatten()

    for cluster_id in range(num_labels):
        # Get indices of the current cluster
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            print(f"Cluster {cluster_id} has no elements, skipping.")
            continue

        print(f"Processing cluster {cluster_id} with {len(indices)} elements")

        # Restrict modes to submesh
        local_modes = W[indices, :]
        sub_volume = volumes[indices]

        # Select modes that are non-zero in the submesh
        is_nonzero = np.any(local_modes != 0, axis=0)
        local_modes = local_modes[:, is_nonzero]
        print (f"Local modes shape after filtering: {local_modes.shape}")

        # Compute basis integrals and sampling matrix
        basis_integrals, A = compute_basis_integrals(sub_volume, local_modes, p)

        if len(basis_integrals) > len(indices):
            print("Warning: More constraints than elements. Using all elements.")
            q_idx = np.arange(len(indices))
            weights = np.ones(len(q_idx))
        else:
            num_samples = min(num_to_sample, len(indices) - 1)
            q_idx, weights, errors = cubature_sampling(A, basis_integrals, num_samples,
                                                       tolerance=tolerance, only_positive=True)

        # Remap to global indices
        cubature_indices.extend(indices[q_idx])
        cubature_weights.extend(weights * sub_volume[q_idx])

    return np.array(cubature_indices), np.array(cubature_weights)

def spectral_moment_fitting_cubature(X, T, W, k, p=2, return_labels=False):
    """
    Generate cubature points using spectral clustering and moment fitting.

    This function averages basis weights onto tetrahedra, clusters them using
    spectral clustering, and then computes cubature within each cluster.

    Parameters:
    - X: Vertex positions (|V| x 3)
    - T: Tetrahedral elements (|T| x 4)
    - W: Per-vertex basis weight matrix (|V| x num_modes)
    - k: Number of clusters to form
    - return_labels: If True, also return the computed cluster labels (default False)

    Returns:
    - cubature_indices: Selected element indices used in cubature
    - cubature_weights: Corresponding cubature weights
    - labels (optional): Cluster label for each tetrahedron
    """
    Wt = average_onto_simplex(W, T)

    [labels, _] = spectral_clustering(Wt, k)
    ret = compute_cluster_cubature(labels, X, T, Wt, p=p)

    if return_labels:
        ret = ret + (labels,)

    return ret

