# Subspace MFEM

Demos for the SIGGRAPH Asia 2023 paper [*Subspace Mixed Finite Elements for Real-Time Heterogeneous Elastodynamics*](https://www.dgp.toronto.edu/projects/subspace-mfem/).

## Requirements

Install `simkit` with the visualization, mesh, learn and video extras from the repo root (see the top-level [README](../../README.md) for the conda setup). At minimum:

```bash
pip install -e ".[mesh,viz,learn,video]"
```

The demos pull meshes from the [`data/`](../../data) submodule, so make sure it's initialised:

```bash
git submodule update --init --recursive
```

## Demos

All three scripts can be run from the repository root.

### Drop test

```bash
python examples/subspace_mfem/drop_fem_vs_mfem.py
```

A heterogeneous-material elastodynamic simulation is run with both a standard FEM solver and our Mixed FEM technique, and the two videos are written to `examples/subspace_mfem/results/drop/` (gitignored).

#### FEM

<div align="center">
  <img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/drop/crab_fem.gif" width="300">
</div>

The crab loses kinetic energy as it falls: the standard Newton solver is slow to converge, so when we cut it short we induce a loss of angular momentum.

#### MFEM

<div align="center">
  <img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/drop/crab_mfem.gif" width="300">
</div>

The Mixed discretization lets the Newton solver be truncated to just a few iterations while preserving angular momentum much better.

### Slingshot test

```bash
python examples/subspace_mfem/slingshot_fem_vs_mfem.py
```

A few vertices of the mesh are pinned, another set is pulled along a specified direction and then released, effectively creating a slingshot.

#### FEM

<div align="center">
  <img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/slingshot/gatorman_fem.gif" width="300">
</div>

The stiff sword resists rotating.

#### MFEM

<div align="center">
  <img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/slingshot/gatorman_mfem.gif" width="300">
</div>

With the mixed formulation, the sword rotates more freely and dynamically.

### Interactive clicking

```bash
python examples/subspace_mfem/interactive_mfem.py
```

Because the method runs in a subspace, the numerical simulation is fast enough to be driven interactively. Right-click anywhere on the mesh to drop a control handle; **hold space** while moving the mouse to drag the most-recently-added handle. Right-clicking elsewhere replaces the active handle.

<div align="center">
<img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/interactive/interactive_crab.gif" width="600">
</div>

## Files

- `drop_fem_vs_mfem.py`, `slingshot_fem_vs_mfem.py`, `interactive_mfem.py` -- the three demos.
- `config.py` -- per-mesh configuration objects (Young's modulus, time step, subspace size, ...).
- `utils.py` -- shared helpers (mesh loading, subspace computation, sim construction, polyscope playback).
