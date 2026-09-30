# Hyper-reduced compliant hand

An SDM-style compliant hand simulated in a [mesh4PDE](https://github.com/otmanon/mesh4PDE) subspace:
Neo-Hookean on a 1,200-vertex coarse mesh, contact and friction on the fine surface, grasping a
ball and a cup. Data in `data/sdm_hand/` (`hand.obj`, `hand_tets.npz`, `hand_rig.npz`, `hand_coarse_1200.npz`).

```bash
python examples/sdm_hand/grasp.py [--full]                  # results/grasp.json, grasp_<case>.png
python examples/sdm_hand/build_hand.py                      # rebuild the fine mesh and rig
MESH4PDE=/path/to/mesh4PDE python examples/sdm_hand/coarsen_hand.py   # rebuild the coarse mesh
```
