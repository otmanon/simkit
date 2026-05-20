"""SimKit: A simulation toolkit for computer animation.

The base install requires only ``numpy`` and ``scipy``. Optional functionality
is enabled by installing extras::

    pip install 'simkit[mesh]'     # libigl       -> mesh utilities
    pip install 'simkit[viz]'      # matplotlib, polyscope
    pip install 'simkit[learn]'    # scikit-learn -> clustering / sampling
    pip install 'simkit[solvers]'  # cvxopt       -> sparse eigensolvers
    pip install 'simkit[video]'    # opencv-python
    pip install 'simkit[cmaes]'    # cma          -> CMA-ES solver
    pip install 'simkit[blender]'  # bpy, blendertoolbox
    pip install 'simkit[all]'      # everything

Importing ``simkit`` is always safe: if an optional dependency is missing, the
affected names simply aren't exported and a single warning is emitted at the
end with concrete install hints.
"""

from __future__ import annotations

import warnings as _warnings

# Maps extra name -> set of dependency module names that failed to import.
_missing: dict[str, set[str]] = {}


class _OptionalImport:
    """Context manager that swallows :class:`ImportError` for optional imports.

    Use it to wrap one ``from .X import Y`` statement. Any ``ImportError``
    raised inside is recorded against ``extra`` and otherwise suppressed, so a
    missing optional dependency just means the corresponding name is not
    exported -- it does not break ``import simkit``.
    """

    __slots__ = ("extra",)

    def __init__(self, extra: str) -> None:
        self.extra = extra

    def __enter__(self) -> "_OptionalImport":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            return False
        if issubclass(exc_type, ImportError):
            _missing.setdefault(self.extra, set()).add(
                getattr(exc_val, "name", None) or "<unknown>"
            )
            return True
        return False


_core = _OptionalImport("all")
_mesh = _OptionalImport("mesh")
_viz = _OptionalImport("viz")
_learn = _OptionalImport("learn")
_solvers = _OptionalImport("solvers")
_video = _OptionalImport("video")
_cmaes = _OptionalImport("cmaes")


# ---------------------------------------------------------------------------
# Pure numpy / scipy (no heavy deps expected, but still guarded for safety).
# ---------------------------------------------------------------------------
with _core:
    from .backtracking_line_search import backtracking_line_search
with _core:
    from .ympr_to_lame import ympr_to_lame
with _core:
    from .normalize_and_center import normalize_and_center
with _core:
    from .orthonormalize import orthonormalize
with _core:
    from .polar_svd import polar_svd
with _core:
    from .stretch_gradient import *  # noqa: F401,F403
with _core:
    from .selection_matrix import selection_matrix
with _core:
    from .symmetric_stretch_map import symmetric_stretch_map
with _core:
    from .dirichlet_penalty import dirichlet_penalty
with _core:
    from .grad import grad
with _core:
    from .project_into_subspace import project_into_subspace
with _core:
    from .pairwise_displacement import pairwise_displacement
with _core:
    from .pairwise_distance import pairwise_distance
with _core:
    from .psd_project import psd_project
with _core:
    from .average_onto_simplex import average_onto_simplex
with _core:
    from .edge_lengths import edge_lengths
with _core:
    from .gradient_cfd import gradient_cfd
with _core:
    from .hessian_cfd import hessian_cfd
with _core:
    from .edge_displacement_jacobian import edge_displacement_jacobian
with _core:
    from .edge_length_jacobian import edge_length_jacobian
with _core:
    from .random_edges import *  # noqa: F401,F403
with _core:
    from .hinge_angles import *  # noqa: F401,F403
with _core:
    from .hinge_hessian import *  # noqa: F401,F403
with _core:
    from .hinge_jacobian import *  # noqa: F401,F403
with _core:
    from .simplex_vertex_map import simplex_vertex_map
with _core:
    from .common_selections import *  # noqa: F401,F403
with _core:
    from .lbs_jacobian import lbs_jacobian

# ---------------------------------------------------------------------------
# libigl-dependent (pip install 'simkit[mesh]').
# ---------------------------------------------------------------------------
with _mesh:
    from .deformation_jacobian import deformation_jacobian
with _mesh:
    from .massmatrix import massmatrix
with _mesh:
    from .volume import volume
with _mesh:
    from .skinning_eigenmodes import skinning_eigenmodes
with _mesh:
    from .cluster_grouping_matrices import cluster_grouping_matrices
with _mesh:
    from .joint_lengths import joint_lengths
with _mesh:
    from .dirichlet_laplacian import dirichlet_laplacian
with _mesh:
    from .linear_modal_analysis import linear_modal_analysis
with _mesh:
    from .shape_outlines import *  # noqa: F401,F403
with _mesh:
    from .stretch import stretch
with _mesh:
    from .subspace_com import subspace_com
with _mesh:
    from .subspace_rotation import subspace_rotation
with _mesh:
    from .gravity_force import gravity_force
with _mesh:
    from .limit_actuation_dirichlet_energy import limit_actuation_dirichlet_energy
with _mesh:
    from .spectral_cubature import spectral_cubature
with _mesh:
    from .rotation_strain_coordinates import rotation_strain_coordinates, RSPrecompute

# ---------------------------------------------------------------------------
# scikit-learn-dependent (pip install 'simkit[learn]').
# ---------------------------------------------------------------------------
with _learn:
    from .farthest_point_sampling import farthest_point_sampling
with _learn:
    from .spectral_clustering import spectral_clustering

# ---------------------------------------------------------------------------
# cvxopt-dependent (pip install 'simkit[solvers]').
# ---------------------------------------------------------------------------
with _solvers:
    from .eigs import eigs

# ---------------------------------------------------------------------------
# Submodules.
# ---------------------------------------------------------------------------
with _mesh:
    from . import sims  # noqa: F401
with _mesh:
    from . import energies  # noqa: F401
with _viz:
    from . import matplotlib  # noqa: F401
with _viz:
    from . import polyscope  # noqa: F401
with _video:
    from . import filesystem  # noqa: F401
with _cmaes:
    from . import solvers  # noqa: F401


# ---------------------------------------------------------------------------
# Aggregated warning -- only fires when something is actually missing.
# ---------------------------------------------------------------------------
def _emit_missing_warning() -> None:
    if not _missing:
        return
    # Surface only categories that map to a real extra, plus an "all" fallback
    # for anything that didn't fit a specific bucket.
    extras = sorted(k for k in _missing.keys() if k != "all")
    lines = []
    for extra in extras:
        deps = ", ".join(sorted(_missing[extra]))
        lines.append(f"  [{extra}] (missing: {deps}) -> pip install 'simkit[{extra}]'")
    if "all" in _missing and _missing["all"]:
        deps = ", ".join(sorted(_missing["all"]))
        lines.append(f"  [other] (missing: {deps}) -> pip install 'simkit[all]'")
    _warnings.warn(
        "simkit: some functionality is unavailable because optional "
        "dependencies are not installed:\n" + "\n".join(lines),
        stacklevel=2,
    )


_emit_missing_warning()
del _emit_missing_warning
