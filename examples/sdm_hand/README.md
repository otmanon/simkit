# Hyper-reduced compliant hand grasping a ball and a cup

A simplified SDM-style compliant hand ([Dollar & Howe, IJRR 2010](https://doi.org/10.1177/0278364909360852)):
stiff polyurethane blocks joined by elastomer flexures, soft fingertip and palm pads,
one actuator spring per joint, all driven by a single actuation parameter `a`. It is
simulated in the subspace of a [mesh4PDE](https://github.com/otmanon/mesh4PDE) coarse mesh:

| Term | Integrated on |
| --- | --- |
| stable Neo-Hookean elasticity | the coarse mesh (1,200 vertices; its tets are the cubature) |
| actuator springs, hinge pins, wrist | embedded points of the fine mesh, pulled back through `x_fine = B x` |
| contact with a rigid object (analytic SDF, cubic penalty) | the fine surface vertices `B_s x` |
| lagged tangential friction | the fine surface vertices in contact |

The coarse mesh is scored through the principal components of a full-space closing
(so the joints keep resolution) mixed with the rest Hessian's eigenmodes (so the pads
do). The flexures sit on the palmar side of the fingers, so closing never drives the
blocks into each other.

## Requirements

```bash
pip install -e ".[all]"
pip install manifold3d tetgen pyvista imageio imageio-ffmpeg scikit-sparse
```

The hand's meshes live in the [`data/`](../../data) submodule (`data/sdm_hand/`):
`hand.obj` (surface), `hand_tets.npz` (tet mesh, part labels, materials), `hand_rig.npz`
(actuators, hinge axes, wrist base, fingertips) and `hand_coarse_1200.npz` (coarse mesh
and prolongation `P`).

## Running

```bash
python examples/sdm_hand/grasp.py            # reduced model: air, ball, cup (about 2 min)
python examples/sdm_hand/grasp.py --full     # + the full space for comparison (about 1 hour)
```

To rebuild the data (offline; `coarsen_hand.py` needs mesh4PDE on the path):

```bash
python examples/sdm_hand/build_hand.py                               # geometry, tet mesh, rig
MESH4PDE=/path/to/mesh4PDE python examples/sdm_hand/coarsen_hand.py  # full-space closing, PCA, coarsening
```

Outputs land in `examples/sdm_hand/results/` (gitignored): `grasp.json` (timings,
Newton iterations, fingertip travel, contact force), `grasp_<case>.png` and `grasp_cup.mp4`.

## Files

| File | Role |
| --- | --- |
| `build_hand.py` | box-CSG geometry, TetGen mesh, winding-number part labels, materials, rig |
| `hand.py` | `build_hand_system` (energy, gradient, Hessian of the full or reduced hand), Newton, actuation sweep |
| `coarsen_hand.py` | PCA + eigenmode basis and the mesh4PDE coarsening |
| `grasp.py` | the experiment: reduced vs full grasps, timings, renders |
