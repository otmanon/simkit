
# Subspace MFEM

This is a python codebase implementing some demos of the contributions proposed by the SIGGRAPH Asia 2023 paper "[Subspace Mixed Finite Elements for Real-Time Heterogeneous Elastodynamics](https://www.dgp.toronto.edu/projects/subspace-mfem/)".


# Installation
First install [simkit](https://github.com/otmanon/simkit.git) a python library containing many of the core simulation utilities upon which these demos are built.
```
git clone --recursive https://github.com/otmanon/simkit.git

cd simkit/
```

It's recommended to use a python environment manager like conda! The code is tested in python 3.10

```
conda create -n subspace_mfem python=3.10
conda activate subspace_mfem
```

Finally install simkit:

```
pip install . 
```

Navigate to the  `\examples\subspace_mfem\` directory.

```
cd \examples\subspace_mfem\
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

<div align="center">
  <img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/drop/crab_fem.gif" width="300">
</div>

Notice how the crab loses a lot of kinetic energy as it falls? This is because the standard Newton solver used to solve the elastodynamic optimization takes forever to converge, so we have to cut it short! It's the cutting it short that induces this loss of angular momentum.

#### MFEM
<div align="center">
  <img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/drop/crab_mfem.gif" width="300">
</div>

In contrast, our Mixed discretization allows the Newton solver to be truncates to few iterations, while much better preserving the angular momentum!



### Slingshot Test

Run 
```
python slingshot_fem_vs_mfem.py
```

By running the above, we show this loss of kinetic energy occurs again even in more involved simulations.
These simulations pin a few parts of the mesh, and pull on another part of the mesh along a specified direction and then lets go, effectively creating a slingshot.


#### FEM
<div align="center">
<img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/slingshot/gatorman_fem.gif" width="300">
</div>

Notice the stiff sword resists rotating.

#### MFEM
<div align="center">
<img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/slingshot/gatorman_mfem.gif" width="300">
</div>

With our mixed formulation, the sword rotates more freely and dynamically!

### Interactive Clicking 

Run 
```
python interactive_mfem.py
```

An important feature of our method is its use of a subspace to carry out simulation. This means that performing the numerical simulation is extremely fast.

To leverage this, the above code runs an interactive simulation, where a user can interactive with a heterogeneous elastic simulation in real-time via control handles.

To add a control handle, **right-click** click anywhere on the mesh.

To drag the last added control handle, **hold the space bar** while hovering your mouse where you want the control handle to move.

Right-clicking on another part of the mesh replaces the previous handle with the new one!

<div align="center">
<img src="https://github.com/otmanon/simkit/blob/main/examples/subspace_mfem/results/interactive/interactive_crab.gif" width="300">
</div>