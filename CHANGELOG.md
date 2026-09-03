# Changelog

All notable changes to SimKit are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays below `1.0`, the public API may change in any release.

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
