# SimKit Tutorials

A hands-on walkthrough of the math inside a deformable-body simulator. Each tutorial is a single Python file you run directly. A window opens with a small 2D (or 3D) scene, a few sliders/buttons, and live energy/iteration plots. You poke the scene with the mouse and watch the numbers move.

The tutorials build on each other: you start by looking at a single triangle, then a static deformable beam, then a beam with mass and gravity, then contact, and finally a 3D block on a floor. Read them in order — concepts introduced earlier are not re-explained later.

## Running a tutorial

From this folder:

```
python 001_deformation_gradient_demo.py
```

You need `simkit`, `numpy`, `scipy`, and `polyscope` installed (see the top-level [README.md](../../README.md)).

## What's in [`utils.py`](utils.py)

The tutorials share a single helper module so each demo can focus on the one idea it's teaching. Three layers:

1. **Mesh / picking primitives** — `triangulated_grid`, `tetrahedralized_grid`, `ball_mesh_2d`, `screen_to_world_2d`, `lame_from_E_nu`.
2. **[`ElasticSim`](utils.py)** — a composable deformable-body simulator. One object owns: state (`U`, `V`, BDF2 history), neo-Hookean material, plus optional pin / handle (mouse-grab) / gravity / contact (plane or sphere). `sim.step(integrator, h)` advances one substep using one of `"Static"`, `"Backward Euler"`, `"BDF2"`, `"Forward Euler"`. Read the class definition once; the tutorials are easier to follow when you know what each demo is *not* doing.
3. **`Viewer2D` / `Viewer3D` / `MouseHandle2D` / `MouseHandle3D`** — polyscope wrappers that handle the boilerplate (registering the mesh, pin/handle markers, click-and-drag picking) so tutorial files stay short.

## Recurring concepts

A few ideas appear in almost every tutorial. Glance at these now and refer back as needed.

- **Rest pose `X` vs. current pose `U`.** `X` is the "undeformed" position of each vertex; `U` is where the vertex is right now. Strain is computed from the map `X → U`.
- **Mesh `T`.** A list of triangles (2D) or tetrahedra (3D). Each row holds the vertex indices for one element.
- **Deformation gradient `F`.** Per-element 2×2 (or 3×3) matrix that describes how a single triangle/tet has been stretched/rotated from rest. `F = I` means no deformation.
- **Energy `E(U)`.** A single scalar measuring how unhappy the material is in its current pose. Forces are `−∇E`. Simulating = repeatedly nudging `U` to reduce `E`.
- **Neo-Hookean energy.** The specific elastic energy used throughout (in [simkit/energies/](../../simkit/energies/)). It penalizes both stretching and volume change and blows up if a triangle inverts.
- **Soft pins (Dirichlet penalty).** Instead of hard-fixing a vertex to a target position, we add a stiff quadratic spring `½ K ‖x − target‖²` to the energy. The pin contributes a constant matrix `Q_pin` and vector `b_pin` to the system — see [simkit/dirichlet_penalty.py](../../simkit/dirichlet_penalty.py).
- **Newton's method.** To minimize `E`, we repeatedly solve `H Δx = −g` (where `g = ∇E`, `H = ∇²E`) and step `x ← x + α Δx`. See [simkit/solvers/NewtonSolver.py](../../simkit/solvers/NewtonSolver.py).

---

## 001 — Deformation gradient: what `F` means

[001_deformation_gradient_demo.py](001_deformation_gradient_demo.py)

A single equilateral triangle. Left-click any vertex and drag it. The triangle's deformation gradient `F` (a 2×2 matrix) prints live, and a rolling plot shows the neo-Hookean elastic energy of the triangle as you deform it.

**What to try:**
- Drag a vertex slightly off rest — `F` is close to the identity, energy is near zero.
- Translate the whole triangle (drag each vertex by the same amount) — `F` stays `I`, energy stays ~0. Rigid motion costs no energy.
- Rotate the triangle — `F` becomes a rotation matrix, energy stays ~0.
- Stretch one edge — diagonal entries of `F` grow, energy grows.
- Try to invert the triangle (flip it inside out) — energy blows up.

**Focus code:**
- `F = deformation_gradient(X, T, U).reshape((2, 2))` is the core mapping from rest+current vertex positions to a per-element matrix.
- `energies.neo_hookean_energy_element_F(F, mu=1, lam=1)` evaluates the strain energy density given that `F`.
- Underlying math lives in [simkit/deformation_gradient.py](../../simkit/deformation_gradient.py) and [simkit/energies/neo_hookean.py](../../simkit/energies/neo_hookean.py).

---

## 002 — Picking and dragging a vertex (warmup)

[002_energy_demo.py](002_energy_demo.py)

Almost the same scene as 001 but reduced to just the interaction layer: right-click selects the nearest vertex, then holding `Space` while moving the mouse drags it.

You can skip this file if you already understood the picking logic in 001.

---

## 003 — Resolution scaling: how mesh size affects solver cost

[003_resolution_complexity.py](003_resolution_complexity.py)

Three cantilever beams — coarse (24 verts), mid (100 verts), and fine (600 verts) — stacked vertically, all using the same physics (BDF2 implicit time integration, Newton solver). A radio button picks which one is currently running under gravity; the other two freeze. Each Newton iteration is wall-clock timed and the per-iteration cost is plotted live for each beam.

**What to learn:**
- Mesh refinement makes the dynamics more accurate but quadratically-to-cubically more expensive per step.
- The bottleneck per Newton iteration is the sparse linear solve `H Δx = −g`, not the energy/gradient assembly.

**Focus code:**
- `newton_step_bdf2` writes out the Newton loop explicitly (instead of going through `ElasticSim.step`) so each iteration can be `time.perf_counter()`-bracketed. The body shows the three Newton-iteration phases: assemble `g`, assemble `H`, `spsolve` + line-search.
- Pieces of the Hessian that don't change between Newton iterations (`pin_H`, kinetic hessian) are cached outside the loop. Only the elastic Hessian is re-built every iteration.

---

## 004 — Newton vs. Gradient Descent: solver choice matters

[004_solver_tradeoff.py](004_solver_tradeoff.py)

A static (no time, no inertia) beam pinned on the left, with one vertex on the right edge soft-pinned to wherever you click. The solver minimizes the total energy each frame; you can swap between Newton and Gradient Descent from a dropdown and change max iterations / line search / step size.

**What to try:**
- With Newton + line search, the beam follows your cursor almost instantly.
- Switch to Gradient Descent, set max iterations to 5 — the beam visibly lags and oozes toward the target.
- Crank GD iterations to 100 — it catches up but each frame is much slower.
- Disable line search — the solver can take wild steps; with GD you may need to lower the step size.

This makes a single point: Newton uses curvature information (`H = ∇²E`) and converges in a handful of iterations near the optimum; gradient descent uses only the gradient and converges much more slowly on this kind of ill-conditioned elastic problem.

**Focus code:**
- `build_solver` constructs either a [NewtonSolver](../../simkit/solvers/NewtonSolver.py) or [GradientDescentSolver](../../simkit/solvers/GradientDescentSolver.py) directly off the `sim.potential_E` / `sim.potential_g` / `sim.potential_H` callables.
- `solver.solve(...)` is the one-line API hiding the optimization loop.

---

## 005 — Integration schemes: Backward Euler vs. BDF2 vs. Forward Euler

[005_integration_scheme.py](005_integration_scheme.py)

A cantilever beam under gravity, dynamic this time. Three integrators sit behind a dropdown: Backward Euler (1st-order implicit), BDF2 (2nd-order implicit), and Forward Euler (explicit). Sliders control the timestep `h` and Young's modulus. Three rolling plots show elastic, kinetic, and total energy.

**What to try:**
- Backward Euler @ `dt = 0.02` — stable, but the beam visibly loses energy over time (numerical damping). Watch the kinetic energy decay even though there is no real damping in the model.
- BDF2 @ `dt = 0.02` — much less damping; the beam keeps oscillating.
- Forward Euler @ `dt = 0.005` — works, looks lively.
- Forward Euler @ `dt = 0.02` — **explodes**. This is the explicit-stability lesson: explicit methods require small timesteps proportional to the stiffest mode of the system.
- Increase Young's modulus with Forward Euler — it explodes at smaller and smaller `dt`.

**Focus code:**
- `sim.step(integrator=..., h=h)` is the only line that differs between integrators. The dispatch and history bookkeeping live in `ElasticSim.step` ([utils.py](utils.py)); the actual kinetic-energy variants live in [simkit/energies/](../../simkit/energies/) under `kinetic_energy_be`, `kinetic_energy_bdf2`, etc.

---

## 006 — Interactive static deformation

[006_interactive_deformation.py](006_interactive_deformation.py)

A beam pinned on the left. Left-click any other vertex and drag — that vertex becomes a soft-pinned handle that follows the cursor; release to let go. The solver minimizes elastic + pin + handle energy each frame, so the beam continuously catches up to wherever you're holding it.

This is the canonical "puppeteer a deformable" interaction loop. Compared to 004, the handle vertex is now arbitrary (instead of a hardcoded vertex on the right edge), and the click-and-drag logic comes from `MouseHandle2D`.

**Focus code:**
- `MouseHandle2D(sim, sel_pc, target_pc)` — wires left-click/drag/release to `sim.grab` / `sim.drag` / `sim.release`. Implementation in [utils.py](utils.py).
- `sim.step(integrator="Static")` does one Newton minimization of the (elastic + pin + handle) potential each frame.

---

## 007 — Interactive dynamics

[007_interactive_dynamics.py](007_interactive_dynamics.py)

Same beam, same handle UI as 006, but now with mass: a kinetic-energy term is added, the integrator is selectable (BE / BDF2 / FE), and `dt` is a slider. Grabbing a vertex while the beam is mid-swing throws inertia around — you can see the beam overshoot, ring, and settle.

This is the merge of 005 (integrators) and 006 (handle UI). Use it to feel the difference between implicit damping (BE settles fast) and energy-preserving behavior (BDF2 keeps ringing).

**Focus code:** identical structure to 006, but `sim.step(...)` now takes an integrator argument other than `"Static"`.

---

## 008 — Contact against a movable ball

[008_interactive_contact_ball.py](008_interactive_contact_ball.py)

A small deformable patch hangs from its top edge. A ball follows your cursor and shoves the patch around through penalty-based contact springs. Sliders expose Young's modulus `E`, Poisson ratio `ν` (via `log10(0.5 − ν)` so you can dial near-incompressible), and the contact stiffness `K`. Press `Space` to delete the ball + reset; left-click to bring it back.

**What to try:**
- Push the ball into the patch slowly — the patch deforms smoothly. Energies plot.
- Increase `K` (contact stiffness) — the patch becomes harder to penetrate; the contact-energy spikes get sharper.
- Lower `K` heavily — the ball starts to clip through the patch (penalty methods are a soft constraint, not a hard one).
- Crank Poisson ratio near `0.5` — the patch resists volume change and behaves more rubber-like.

**Focus code:**
- `sim.set_sphere_contact(K=..., center=ball_p, radius=BALL_R)` enables one-sided contact springs that activate when a vertex penetrates the sphere. `sim.update_sphere_center(p)` re-targets the ball each frame.
- `sim.contact_E(x)` plots the contact penalty; under the hood it calls [`contact_springs_sphere_energy`](../../simkit/energies/contact_springs_sphere.py).

---

## 009 — Contact against a static floor (2D)

[009_interactive_contact_plane.py](009_interactive_contact_plane.py)

A patch sits above a horizontal floor. Gravity pulls it down; the floor pushes it up via plane-contact penalty springs. Left-click + drag any vertex to throw the patch around, then release to let it fall back.

**What to try:**
- Just let it sit — it should rest on the floor with a small visible bulge where vertices are slightly below the floor (penalty methods always allow some penetration).
- Grab a corner and lift the patch high, then drop it — watch it bounce and settle.
- Switch to Forward Euler and slowly increase `dt` — the contact spring becomes the stiffest mode and is the first thing to explode.

**Focus code:**
- `sim.set_plane_contact(K=..., p=[0, floor_y], n=[0, 1])` swaps in plane contact. Implementation: [`contact_springs_plane_energy`](../../simkit/energies/contact_springs_plane.py).
- The rest of the file is identical in shape to 007, modulo the handle vertex now being free to move below the floor (it tugs the patch through the floor with sufficient force).

---

## 010 — Contact in 3D

[010_interactive_contact_plane_3D.py](010_interactive_contact_plane_3D.py)

The 2D scene from 009 lifted into 3D: a tetrahedral block dropped onto a floor. Same neo-Hookean elasticity, same plane-penalty contact, same integrators. The mesh is built from a hex grid that splits each cell into five tets (see `tetrahedralized_grid` in [utils.py](utils.py)).

**What to try / how to interact:**
- The mouse owns two modes. By default, the "Click-to-drag handle" checkbox is **on** — left-click a vertex and drag to deform the block in 3D. To orbit/zoom the camera, **uncheck** the box (polyscope takes the mouse back).
- Picking in 3D uses polyscope's `screen_coords_to_world_position` to find the world point under the cursor, then snaps to the nearest mesh vertex. Dragging keeps the handle on a camera-facing plane through the picked point — see `MouseHandle3D` in [utils.py](utils.py).
- Crank Young's modulus high — the block barely deforms on impact. Lower it — the block squishes flat and slowly recovers.

**Focus code:**
- Almost everything is identical to 009 — the `ElasticSim` API is dimension-agnostic. The only changes are (a) `Viewer3D` instead of `Viewer2D`, (b) `MouseHandle3D` instead of `MouseHandle2D`, (c) `tetrahedralized_grid` instead of `triangulated_grid`.

---

## Where to go next

The tutorials cover the spine of an FEM simulator: deformation gradient → energy → static solve → dynamic solve → contact → 3D. Once these feel comfortable, look at:

- [simkit/energies/](../../simkit/energies/) — the actual neo-Hookean / kinetic / contact energy implementations, including their analytic gradients and PSD-projected Hessians.
- [simkit/solvers/](../../simkit/solvers/) — `NewtonSolver`, `GradientDescentSolver`, and the line-search and backtracking utilities.
- [examples/](..) — larger end-to-end scenes that compose these same primitives.
