import numpy as np
import scipy.sparse as sps

import simkit
from simkit.energies import (sdf_contact_energy_x, sdf_contact_gradient_x, tangential_friction_energy_x,
                             tangential_friction_gradient_x, tangential_friction_hessian_x)


def test_embedding_sdfs_contact_and_friction() -> None:
    rng = np.random.default_rng(0)
    X, Q = rng.standard_normal((40, 3)), rng.standard_normal((3, 3))
    G = simkit.affine_embedding_matrix(X, Q, [np.arange(40)] * 3, k=12)
    A, t = rng.standard_normal((3, 3)), rng.standard_normal(3)
    assert np.allclose((G @ (X @ A.T + t).ravel()).reshape(-1, 3), Q @ A.T + t)
    assert np.allclose(simkit.cup_sdf(np.array([[0, 0, 1.5], [0, 0, 0.0]]), np.zeros(3), 1, 0.1, 0.2, 2)[0][0], 0.5)
    x, S = 0.9 * rng.standard_normal(24), sps.random(18, 24, density=0.3, random_state=2, format="csr")
    a, n, p0 = rng.random(6), rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    sdf = lambda P: simkit.sphere_sdf(P, np.zeros(3), 1.0)
    for E, g in [(lambda y: sdf_contact_energy_x(y, S, sdf, 3.0, a), sdf_contact_gradient_x(x, S, sdf, 3.0, a)),
                 (lambda y: tangential_friction_energy_x(y, S, p0, n, a), tangential_friction_gradient_x(x, S, p0, n, a))]:
        fd = simkit.gradient_cfd(lambda y: np.array([E(y)]), x.reshape(-1, 1), 1e-6)
        assert np.allclose(g.ravel(), fd.ravel(), atol=1e-6)
    v, H = rng.standard_normal(24), tangential_friction_hessian_x(x, S, p0, n, a)
    E = lambda y: tangential_friction_energy_x(y, S, p0, n, a)
    assert np.isclose(E(x + v), E(x) + tangential_friction_gradient_x(x, S, p0, n, a).ravel() @ v + 0.5 * v @ (H @ v))
