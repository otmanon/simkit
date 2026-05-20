
# SimKit : A Simulation Toolkit for Computer Animation

[![CI](https://github.com/otmanon/simkit/actions/workflows/ci.yml/badge.svg)](https://github.com/otmanon/simkit/actions/workflows/ci.yml)
[![Docs](https://github.com/otmanon/simkit/actions/workflows/docs.yml/badge.svg)](https://github.com/otmanon/simkit/actions/workflows/docs.yml)

This library should be considered a toolbox for the development of physically-based animation research.
It is designed to be modular, easy to use, and easy to extend.
In particular, it is designed to be emphasize fast creative and experimental prototyping.


## Installation

Clone the repository:
```bash
git clone --recursive https://github.com/otmanon/simkit.git
```

Installation is recommended on a fresh conda environment:

```bash
cd simkit
conda create -n simkit python=3.11
conda activate simkit
pip install -e .
```

## Optional Dependencies

The base install only requires `numpy` and `scipy`. Heavier or specialized
dependencies are exposed as named extras so you only install what you need.
Importing `simkit` is always safe -- functionality whose extras are missing
just isn't exported, and a one-line warning tells you exactly what to install.

| Extra      | Adds                                       | Enables                                                          |
| ---------- | ------------------------------------------ | ---------------------------------------------------------------- |
| `mesh`     | `libigl`                                   | Core mesh ops: `deformation_jacobian`, `massmatrix`, `volume`, ... |
| `viz`      | `matplotlib`, `polyscope`, `libigl`        | `simkit.matplotlib`, `simkit.polyscope` plotters                 |
| `learn`    | `scikit-learn`                             | `farthest_point_sampling`, `spectral_clustering`                 |
| `solvers`  | `cvxopt`                                   | Sparse eigensolvers (`simkit.eigs`)                              |
| `video`    | `opencv-python`                            | `simkit.filesystem` GIF / video helpers                          |
| `cmaes`    | `cma`                                      | `simkit.solvers.CMAESSolver`                                     |
| `blender`  | `bpy`, `blendertoolbox`                    | Blender rendering helpers                                        |
| `all`      | union of the above                         | Everything end-user-facing                                       |
| `dev`      | `pytest`, `pytest-cov`                     | Running the test suite                                           |
| `docs`     | `sphinx`, `sphinx-autoapi`, `furo`, ...    | Building the documentation                                       |

Install one or more extras with the usual `pip` syntax:

```bash
pip install -e ".[mesh]"            # just mesh ops
pip install -e ".[mesh,viz,learn]"  # multiple extras
pip install -e ".[all]"             # everything end-user-facing
pip install -e ".[all,dev,docs]"    # everything, including dev tooling
```

## Running Examples

The repository includes several example scripts demonstrating different simulation capabilities:

```bash
cd examples
python run_elastics_reduced.py
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Building the documentation

The docs are generated from the docstrings in the `simkit` package using
Sphinx + `sphinx-autoapi`. To build them locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in your browser. Every function in
`simkit/` is automatically picked up and rendered into the API reference --
just write a good docstring and rebuild.
