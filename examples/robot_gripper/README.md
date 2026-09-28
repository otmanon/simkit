# Robot gripper with soft pads grasping a fragile cup

A parallel-jaw electric gripper (Robotiq Hand-E / Schunk EGP class) with silicone
fingertip pads, plus a thin-walled polystyrene cup.

| step | script | output (`examples/robot_gripper/output/`) |
| --- | --- | --- |
| 1. parametric CSG geometry (manifold3d) | `gripper_geometry.py` | `parts/*.obj`, unified `gripper.obj`, `cup.obj` |
| 2. TetGen + libigl winding-number material labels | `tetrahedralize_and_label.py` | `scene_tets.npz` (per-tet part, E, ν, ρ, strength) |
| 3. render material parameters | `render_materials.py` | `renders/*.png` |
| 4. grasp + lift simulation (SimKit) | `simulate_grasp.py` | `grasp_frames.npz` |

Materials (`materials.py`): Al 6061-T6 housing/carriages, Al 7075-T6 fingers,
440C steel guide rail, Shore 10A silicone pads, GPPS polystyrene cup.

Actuation in `simulate_grasp.py`: the housing follows the robot flange (lift),
the jaw carriages (found by winding number) slide along the rail with a
prescribed stroke, and fingers, pads and cup are free elastic bodies. Pads and
cup interact through penalty contact with lagged Coulomb friction.

Extra dependencies: `pip install manifold3d tetgen pyvista libigl`
(optional: `scikit-sparse` for CHOLMOD).
