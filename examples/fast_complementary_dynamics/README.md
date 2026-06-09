# Fast Complementary Dynamics

A minimal slice of [*Fast Complementary Dynamics via Skinning Eigenmodes*](https://www.dgp.toronto.edu/projects/fast-complementary-dynamics/) (Benchekroun et al., SIGGRAPH 2023).

A 2D beam is driven by an animator rig (a single global handle) and that handle is swept left/right. Two simulators are exercised back-to-back:

| World | What it does |
| --- | --- |
| `CoDyLeftRightWorld` | Fast Complementary Dynamics. The simulation subspace is constrained to be orthogonal to the rig, so secondary motion is purely complementary. |
| `PinLeftRightWorld` | A baseline that hard-pins a small region in the middle of the beam and sweeps the pinned vertices left/right. |

## Requirements

This demo uses matplotlib for rendering, plus Pillow and a system `ffmpeg` for `.mp4` / `.gif` export. From the repo root:

```bash
pip install -e ".[viz,video]"
pip install gpytoolbox
```

`gpytoolbox` is only used to generate the demo's tiny rectangular mesh and is not a hard dependency of `simkit`.

## Running

```bash
python examples/fast_complementary_dynamics/left_right_control.py
```

Resulting videos and gifs land in `examples/fast_complementary_dynamics/results/` (gitignored). Two outputs are produced:

- `beam_cody_left_right.mp4` / `.gif` -- complementary dynamics
- `beam_pin_0.1_left_right.mp4` / `.gif` -- pinned baseline

## Customising

`left_right_control.py` exposes both worlds as classes; build the mesh you want (any `(X, T)`), wrap it in `CoDyLeftRightWorld(X, T)` or `PinLeftRightWorld(X, T, bI=...)`, and call `.simulate_periodic_x(...)` followed by `.render(...)`. See the source for the small set of knobs.
