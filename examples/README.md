# SimKit Examples

End-to-end demos that exercise the library on small reproductions of papers
that motivated `simkit`. Each demo lives in its own subdirectory with a
self-contained README.

## Index

| Demo | Paper | Highlights |
| --- | --- | --- |
| [`fast_complementary_dynamics/`](fast_complementary_dynamics) | [Fast Complementary Dynamics via Skinning Eigenmodes (SIGGRAPH 2023)](https://www.dgp.toronto.edu/projects/fast-complementary-dynamics/) | Rig-driven 2D beam with complementary secondary motion vs. a pinned baseline. |
| [`modal_muscles/`](modal_muscles) | Modal Muscles / Actuators a la Mode | 2D creature whose modal-actuator parameters are optimised by CMA-ES to walk forward. |
| [`subspace_mfem/`](subspace_mfem) | [Subspace Mixed FEM for Real-Time Heterogeneous Elastodynamics (SIGGRAPH Asia 2023)](https://www.dgp.toronto.edu/projects/subspace-mfem/) | FEM vs. MFEM drop and slingshot tests, plus an interactive click-to-pull demo. |
| [`force_dual_modes/`](force_dual_modes) | Force-dual modes / linear modal analysis subspaces | _Source not yet on `main` -- see the [`force_dual_modes`](https://github.com/otmanon/simkit/tree/force_dual_modes/examples/force_dual_modes) branch._ |

## Running a demo

All demos are standalone scripts -- they assume `simkit` is installed (e.g. `pip install -e ".[mesh,viz,learn,video]"` from the repo root) and can be invoked directly:

```bash
python examples/<demo>/<script>.py
```

Most demos also rely on the [`data/`](../data) submodule for meshes, so make sure it's checked out:

```bash
git submodule update --init --recursive
```

## Conventions

- Each demo writes generated artifacts (videos, gifs, cached subspaces, optimisation histories, ...) to `examples/<demo>/results/`. That directory is gitignored -- nothing under it should be committed.
- Scripts use `os.path.dirname(__file__)` to locate data and write outputs, so they can be run from any working directory.
- Optional simkit features (matplotlib, polyscope, libigl, scikit-learn, cma, opencv-python, ...) are pulled in via the `simkit` extras documented in the [root README](../README.md). Each demo's README lists the extras it actually needs.
