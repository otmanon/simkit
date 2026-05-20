
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
| `docs`     | `sphinx`, `sphinx-autoapi`, `pydata-sphinx-theme`, ... | Building the documentation                            |

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
Sphinx + `sphinx-autoapi` + the PyData theme. To build them locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in your browser. Every function in
`simkit/` is automatically picked up and rendered into the API reference --
just write a good docstring and rebuild.

## Release / dev workflow

Two equivalent ways to do every build/test/release/docs task. Pick whichever
you prefer.

- **`scripts/COMMANDS.md`** — raw, copy/paste-able bash commands organized by
  task. Read it top-to-bottom or grab whichever block you need. No
  abstraction, no functions, just shell commands.
- **`scripts/release.sh`** — the same commands wrapped as subcommands so you
  can run e.g. `./scripts/release.sh build` instead of pasting. There's also
  a `Makefile` with `make build` / `make docs` / etc. targets.

First-time setup on a new machine:

```bash
chmod +x scripts/release.sh
conda activate simkit   # so `python` points at the right interpreter
```

Quick reference for the script form:

| What | Command |
| --- | --- |
| Build sdist + wheel | `./scripts/release.sh build` (or `make build`) |
| Upload to TestPyPI | `./scripts/release.sh upload-test` |
| Install from TestPyPI in a throwaway venv | `./scripts/release.sh test-install` |
| Upload to **real** PyPI | `./scripts/release.sh upload-prod` |
| Build the docs | `./scripts/release.sh docs` |
| Build + open the docs | `./scripts/release.sh docs-open` |
| Remove all build/docs/cache junk | `./scripts/release.sh clean` |

### Local upload vs GitHub Actions Trusted Publishing

The script has the full tradeoff in a header comment. Summarized:

- **Local** (`upload-test` / `upload-prod`): fastest path, but requires a
  PyPI API token on your laptop and the build runs in your local
  environment. Best for the very first upload (claiming the project name)
  or one-off hotfixes.
- **GitHub Actions** (`.github/workflows/release.yml`, triggered by pushing
  a `vX.Y.Z` git tag): no secrets stored anywhere thanks to Trusted
  Publishing; every release is a reproducible build from a tagged commit;
  auto-publishes to TestPyPI first, then PyPI. Best for every release
  after the first.

Recommended flow: claim the project on PyPI/TestPyPI with one local upload,
then switch to tag-triggered GitHub Actions for everything afterward:

```bash
git tag v0.1.0
git push --tags
```
