# Changelog

All notable changes to SimKit are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays below `1.0`, the public API may change in any release.

## [Unreleased]

### Added

- **C++ core with Python bindings** (`cpp/`, `simkit._cpp`). The slow NumPy
  paths now have Eigen implementations behind the same flat functions:
  every `*_element_F` material kernel (`arap`, `fcr`, `neo_hookean`,
  `stable_neo_hookean`, `stvk`, `linear_elasticity`,
  `macklin_mueller_neo_hookean` and the `elastic_*` dispatcher),
  `polar_svd`, `svd_rv`, `rotation_gradient_F`, `psd_project`,
  `deformation_jacobian`, and the Hessian assembly. The extension is built
  by `pip install .` when a C++17 compiler is available and is optional:
  without it the package is unchanged. Results agree with the NumPy code to
  round-off (`tests/test_cpp_backend.py`).
- **`block_diag_sandwich(J, He)`**: `J.T @ block_diag(He) @ J` without
  materializing the block diagonal. All `*_hessian_x/_u/_z` functions now
  assemble through it (sparse or dense `J`).
- **Backend helpers** `backend()`, `set_backend()`, `cpp_available()`,
  `cpp_num_threads()` and the `SIMKIT_BACKEND` / `SIMKIT_NUM_THREADS` /
  `SIMKIT_NO_CPP` environment variables.
- `scripts/benchmark_cpp.py` to time both backends, and a standalone CMake
  build with C++ tests under `cpp/`.
- **Assembled tiers in C++**: `elastic_{energy,gradient,hessian}_x` fused
  (`F = J x`, element kernel, `vol` weighting, PSD projection, CSR assembly)
  behind every per-material `*_x` / `*_u` function and the dispatcher, and
  the subspace `elastic_*_z` tiers with a dense reduced operator.
- `examples/benchmarks/cpp_backend_speedup.ipynb`: the speedup experiment
  for `svd_rv`, the neo-Hookean element terms, the assembled and subspace
  gradient / Hessian, individually and together.

- **Assembly as its own functions**: `assembled_gradient(J, P)`,
  `assembled_hessian(J, He)` (full space, sparse `J`) and
  `assembled_subspace_gradient(JB, P)`, `assembled_subspace_hessian(JB, He)`
  (dense reduced operator), in C++ with the NumPy products as fallback. The
  dense ones are chunked over elements across the worker threads, and the
  subspace Hessian uses a symmetric rank update (half the flops) whenever
  the blocks are positive definite. The reduced `elastic_*_z` tiers and
  `block_diag_sandwich` assemble through them.
- CI builds and tests the extension on Linux, macOS and Windows.
- **`sparse_solve(A, b)`**: direct solve on the C++ backend with Eigen's
  `SimplicialLLT` (Cholesky), `SimplicialLDLT` if the system is not positive
  definite, and `scipy.sparse.linalg.spsolve` as the last resort; the symbolic
  analysis is reused across calls with the same sparsity pattern.
  `newton_solver` uses it, which roughly halves the linear-solve time of a
  Newton step. `return_method=True` reports which solver ran.
- **`read_mesh(path)`**: NumPy-only reader for MEDIT `.mesh` files
  (vertices, triangles, tetrahedra), so the `data/` meshes load without libigl.
- `examples/interactive_demos/016_octopus_grab_cpp_vs_python.py`: click-and-drag
  soft handles on a pinned neo-Hookean octopus with a live C++ / Python
  backend switch (and a `--headless` timing mode).

### Fixed

- `elastic_energy_filtered_z` raised a `TypeError` on NumPy 2 when reducing
  the quadratic filter term to a float.

## [0.1.8] - 2026-09-23

### Removed

- **`filesystem.notebook_results_path`** (added in 0.1.7). Per-notebook results
  folders are a tutorials concern, not a library one; the helper now lives in
  the simkit-tutorials repo's `utils.py`. `save_figure` and `save_animation`
  are unchanged.

## [0.1.7] - 2026-09-23

### Added

- **`filesystem.save_figure`** — saves a Matplotlib figure, creating parent
  folders, and optionally copies it to extra destinations (e.g. a tracked
  `media/` folder).
- **`filesystem.save_animation`** — saves a Matplotlib animation to `.mp4`
  (ffmpeg) or `.gif` (Pillow) with the same copy-to behaviour. Matplotlib's
  writers are imported only when called, so `simkit.filesystem` still imports
  on a base install.
- **`filesystem.notebook_results_path`** — returns (and creates)
  `results/<notebook_name>/` (removed again in 0.1.8).

## [0.1.6] - 2026-09-03

A cleanup release: no new numerics, but several public functions that were
unreachable or silently wrong are now correct, and the test suite grew from
475 to 622 tests.

### Fixed

- **`membrane_deformation_jacobian` exported the wrong implementation.** The
  name resolved to a copy inside `deformation_jacobian.py` that built the
  volumetric `H` and raised `ValueError` on triangles-in-3D — the only input it
  was documented for. Membrane is now its own module, and the exported function
  returns the correct `3x2` operator that `membrane_neo_hookean_*` expects.
- **31 public functions were shadowed by their own modules.** Without an
  explicit re-export, `simkit.triangle_areas` resolved to the *module*, so
  calling it raised `TypeError: 'module' object is not callable`. All are now
  exported (see Added).
- **`solvers.cmaes()` raised `UnboundLocalError` on its default arguments.**
  The single-process branch appended to `running_history` unconditionally,
  though that list is only created when `return_history=True`.
- **`filesystem.compute_with_cache_check` corrupted length-1 arrays.** Loading
  from cache called `.item()` on every stored value, so a `np.array([7.0])`
  written on a cache miss came back as the float `7.0` on the next hit. Only
  0-d arrays (the pickled non-array objects) are unwrapped now.
- **`simkit.filesystem` was gated behind the `[video]` extra**, making
  `get_data_directory`, `compute_with_cache_check` and `mp4_to_gif` unavailable
  on a base install even though none of them need Pillow. Only
  `video_from_image_dir` is gated now.
- **`simkit.solvers` was imported inside the `[cmaes]` optional bucket**, so a
  missing `cma` removed *every* solver. `solvers/cmaes.py` guards its own
  import, so the bucket was dead code; it has been removed.
- Malformed reStructuredText in docstrings that broke the rendered API
  reference: an unparseable table in `discrete_shells_bending`, math blocks
  whose `+`/`-` continuation lines rendered as bullet lists (`neo_hookean`,
  `stable_neo_hookean`, `stvk`, `macklin_mueller_neo_hookean`), `|F|`-style
  substitution errors (`edge_face_adjacency`, `harmonic_coordinates`), and
  stray `*` emphasis (`kinetic`, `simplex_vertex_map`,
  `rotation_strain_coordinates`).

### Added

- `energies.elastic`'s `material=` dispatcher now covers `'stvk'`,
  `'neo-hookean'` and `'stable-neo-hookean'` in addition to the previous four,
  across every tier. `emu` (parameterized by fibre direction and activation)
  and `membrane_neo_hookean` (non-square `F`) remain outside it by design and
  raise `ValueError`; both are documented in the module docstring.
- Top-level exports for 31 previously shadowed functions, including
  `boundary_edges`, `edges`, `normals`, `triangle_areas`, `tetrahedron_volumes`,
  `svd_rv`, `vectorized_trace`, `vectorized_transpose`, `interweaving_matrix`,
  `harmonic_coordinates`, `biharmonic_coordinates` and `eigs_iccm` (the last
  behind the `[solvers]` extra).
- `simkit.energies`, `simkit.solvers` and `simkit.filesystem` are now imported
  explicitly at the top level rather than as a side effect of another import.
- Tests for solvers, integrators, filesystem, P2 elements, mesh topology,
  matplotlib and polyscope helpers, the elastic dispatcher, and the shape of
  the public API — 475 to 622 tests.
- Package docstring for `simkit.energies` recording the tier convention and
  which tiers are deliberately partial.

### Changed

- **`solvers.local_global` raises `NotImplementedError`** instead of returning
  `x + 1`. It never had an implementation; use `solvers.block_coord`, which
  performs the same local/global alternation and is tested.
- `newton_solver`'s `max_iter=1` default is now documented explicitly: it is a
  per-timestep *stepper*, not a solve to convergence. Pass `max_iter` when
  using it as a solver. The default itself is unchanged.
- `arap`'s reference docstring documents the `_u` displacement tier, which
  existed everywhere but was missing from the convention it defines.

### Removed

- `tests/conftest.py`, which installed a stub over `simkit.energies` to work
  around a refactor that finished long ago. The aggregate package is now
  actually exercised by the suite.

[0.1.6]: https://github.com/otmanon/simkit/compare/v0.1.5...v0.1.6
