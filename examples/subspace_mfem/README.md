
# Subspace MFEM

This is a python codebase implementing some demos of the contributions proposed by the SIGGRAPH Asia 2023 paper "[Subspace Mixed Finite Elements for Real-Time Heterogeneous Elastodynamics](https://www.dgp.toronto.edu/projects/subspace-mfem/)".


# Installation
First install [simkit](https://github.com/otmanon/simkit.git) a python library containing many of the core simulation utilities upon which these demos are built.
```
git clone --recursive https://github.com/otmanon/simkit.git

cd simkit/examples/subspace_mfem/
```


It's recommended to use a python environment manager like conda! The code is tested in python 3.10

```
conda create -n subspace_mfem python=3.10
conda activate subspace_mfem
```

Be sure to install all the dependencies for simkit!

```
pip install numpy scipy libigl polyscope scikit-learn matplotlib cvxopt opencv-python
```

# Demos
Once inside the `simkit\examples\subspace_mfem\` repository, you will be met with a handful of demos in the format of `*.py` files!

### Drop Test
Run 
```
python drop_fem_vs_mfem.py
```

A simulation of a heterogeneous material elastodynamic simulation will then be run with both standard FEM, and our Mixed FEM technique!
It will output two videos in `\results\drop\` :

#### FEM
 <iframe width="560" height="315" src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/drop/crab_fem.mp4" frameborder="0" allowfullscreen></iframe>
#### MFEM

### Slingshot Test

### Interactive Clicking 

### Visualising Subspace
