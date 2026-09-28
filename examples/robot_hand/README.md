# Allegro robot hand grasping a fragile cup

Same pipeline as `../robot_gripper`, but with a real research robot hand: the
**Wonik Allegro Hand v3** (16 DoF, 4 fingers). The link meshes and
kinematics are the MuJoCo Menagerie model (`wonik_allegro/`, BSD-2-Clause,
vendored with its LICENSE).

| step | script | output (`output/`) |
| --- | --- | --- |
| 1. posed CAD → solids: knuckle clearances (Minkowski), steel joint shafts, rubber tips over Al cores | `hand_geometry.py` | `parts/*.obj`, unified `hand.obj` (17 shells, one per link), `cup.obj`, `hand_meta.json` |
| 2. fTetWild per shell + libigl winding-number labels | `tetrahedralize_and_label.py` | `scene_tets.npz` (per-tet part, rigid body, E, ν, ρ) |
| 3. fit a grasp within joint limits (FK) | `hand_grasp.py` | `grasp_q` in `hand_meta.json` |
| 4. render materials | `render_hand.py [--open]` | `renders/*.png` |
| 5. close + lift (SimKit FEM) | `simulate_hand.py [--steps N]` | `hand_frames.npz` |

`allegro_kinematics.py` is a dependency-free MJCF parser + forward kinematics.

Materials (`materials.py`): Al 6061-T6 palm and tip cores, Al 7075-T6
phalanges, AISI 4140 steel joint shafts, polyurethane 40A fingertip rubber,
GPPS polystyrene cup.

Actuation: joint trajectory `q(t)` (flat → grasp) drives the palm and the
steel joint shafts through forward kinematics, the way the Allegro's motors
act on each link. Phalanges, rubber tips and cup are free elastic bodies;
tips and cup interact through penalty contact with lagged Coulomb friction.

TetGen could not handle the Allegro CAD surfaces (internal errors or 400k+
tets per link), hence fTetWild. Extra dependencies:
`pip install manifold3d wildmeshing tetgen pyvista libigl trimesh`.
