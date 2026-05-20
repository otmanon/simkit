# Installation

## From source (recommended for now)

```bash
git clone --recursive https://github.com/otmanon/simkit.git
cd simkit
conda create -n simkit python=3.11
conda activate simkit
pip install -e .
```

## Optional extras

The base install requires only `numpy` and `scipy`. Heavier dependencies are
opt-in:

```bash
pip install -e ".[mesh]"     # libigl       -> core mesh utilities
pip install -e ".[viz]"      # matplotlib, polyscope (also pulls libigl)
pip install -e ".[learn]"    # scikit-learn -> clustering, sampling
pip install -e ".[solvers]"  # cvxopt       -> sparse eigensolvers
pip install -e ".[video]"    # opencv-python
pip install -e ".[cmaes]"    # cma          -> CMA-ES solver
pip install -e ".[blender]"  # bpy, blendertoolbox
pip install -e ".[all]"      # everything end-user-facing
pip install -e ".[dev]"      # pytest + coverage
pip install -e ".[docs]"     # Sphinx tooling
```

Combine multiple extras with commas, e.g. `pip install -e ".[mesh,viz,learn]"`.

Importing `simkit` is always safe -- if an optional dependency is missing the
affected names simply aren't exported and a one-line warning shows you what to
install.

## Building the docs locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.
