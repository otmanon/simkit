# Modal Muscles

A 2D locomotion demo in the spirit of *Modal Muscles / Actuators a la Mode*.

A reduced elastic creature is actuated by a tiny number of *modal* muscles -- a small set of linear-modal-analysis directions that are driven by sinusoids. A few sinusoid parameters (amplitude, period, phase) are tuned with CMA-ES so the centre of mass moves forward and the body stays upright.

## Requirements

```bash
pip install -e ".[mesh,viz,learn,cmaes,video]"
```

The demo also assumes the [`data/`](../../data) submodule is initialised (it provides the 2D `horse.obj` mesh):

```bash
git submodule update --init --recursive
```

## Running

```bash
python examples/modal_muscles/actuators_a_la_mode_2D.py
```

The CMA-ES run takes a few minutes (default 200 iterations, population 16, 8 processes). After it finishes, a matplotlib window opens showing the optimised gait played back through `animation_viewer_2D`.

## Outputs

All intermediate state is written to `examples/modal_muscles/results/2d/horse/` (gitignored):

- `subspace.npz` -- cached skinning eigenmodes / modal analysis / cubature
- `result.npz` -- optimised parameters + CMA-ES running history
- `fbest_history.png` -- best objective per CMA-ES generation

Re-running the script picks up the cached files via `simkit.filesystem.compute_with_cache_check`. Set `read_cache = False` at the top of the script to force recomputation.

## Files

- `actuators_a_la_mode_2D.py` -- end-to-end pipeline (mesh -> subspace -> sim -> CMA-ES -> viewer).
- `animation_viewers.py` -- standalone matplotlib viewer used at the end of the script.

## Switching mesh

Change `name = "horse"` near the top of `actuators_a_la_mode_2D.py` to any other mesh that lives under `data/2d/<name>/<name>.obj`.
