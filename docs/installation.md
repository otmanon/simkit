# Installation

SimKit is on PyPI and requires Python 3.10+.

## With pip

```bash
pip install simkit
```

Installing into a fresh environment is recommended:

```bash
conda create -n simkit python=3.11
conda activate simkit
pip install simkit
```

Or with the standard library's `venv`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install simkit
```

## With uv

[uv](https://docs.astral.sh/uv/) is a fast drop-in replacement for `pip` and
`venv`. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

To add SimKit to an existing uv-managed project:

```bash
uv add simkit
uv add "simkit[viz]"          # with extras
```

To install into a plain virtual environment with `uv pip`:

```bash
uv venv --python 3.11
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install simkit
uv pip install "simkit[all]"  # with extras
```

To try SimKit without installing anything permanently:

```bash
uv run --with "simkit[viz]" python my_script.py
```

## Optional extras

The base install requires only `numpy` and `scipy`. Heavier dependencies are
opt-in:

```bash
pip install "simkit[mesh]"     # libigl       -> 2D Triangle meshing
pip install "simkit[viz]"      # matplotlib, polyscope
pip install "simkit[solvers]"  # cvxopt       -> sparse eigensolvers
pip install "simkit[video]"    # Pillow       -> image/video frame helpers
pip install "simkit[cmaes]"    # cma          -> CMA-ES solver
pip install "simkit[all]"      # everything end-user-facing
pip install "simkit[dev]"      # pytest + coverage
pip install "simkit[docs]"     # Sphinx tooling
```

Combine multiple extras with commas, e.g. `pip install "simkit[mesh,viz]"`.
The same syntax works with `uv pip install` and `uv add`.

Importing `simkit` is always safe -- if an optional dependency is missing the
affected names simply aren't exported and a one-line warning shows you what to
install.

## From source

For development, or to run the demos under `examples/`:

```bash
git clone --recursive https://github.com/otmanon/simkit.git
cd simkit
conda create -n simkit python=3.11
conda activate simkit
pip install -e ".[all,dev]"
```

The editable install works the same with uv:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

## Building the docs locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.
