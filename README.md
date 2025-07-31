
# SimKit : A Simulation Toolkit for Computer Animation

This library should be considered a toolbox for the development of physically-based animation research.
It is designed to be modular, easy to use, and easy to extend.
In particular, it is designed to be emphasize fast creative and experimental prototyping.


## Installation

Clone the repository:
```bash
git clone --recursive https://github.com/otmanon/simkit.git
```

Installation is recommended on a fresh conda directory:

```bash
cd simkit
conda create -n simkit python=3.10
conda activate simkit
pip install -e .
```

## Optional Dependencies

Some features require additional packages that are not installed by default. You can install these optional dependencies as needed:

### CMAES Solver
The `CMAESSolver` requires the `cma` package for CMA-ES optimization:

```bash
# Install with CMAES support
pip install -e .[cmaes]
```

### All Optional Dependencies
To install all optional dependencies at once:

```bash
pip install -e .[all]
```

## Running Examples

The repository includes several example scripts demonstrating different simulation capabilities. To run an example:

```bash
cd examples
python run_elastics_reduced.py
```

