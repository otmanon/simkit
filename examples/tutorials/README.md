# SimKit Tutorials

A hands-on walkthrough of the math inside a deformable-body simulator. Each tutorial is a single Python file you run directly. A window opens with a small 2D (or 3D) scene, a few sliders/buttons, and live energy/iteration plots. You poke the scene with the mouse and watch the numbers move.

The tutorials build on each other: you start by looking at a single triangle, then a static deformable beam, then a beam with mass and gravity, then contact, and finally a 3D block on a floor. Read them in order — concepts introduced earlier are not re-explained later.

## Running a tutorial

From this folder:

```
python 001_deformation_gradient_demo.py
```

You need `simkit`, `numpy`, `scipy`, and `polyscope` installed (see the top-level [README.md](../../README.md)). The `utils.py` file in this folder provides small helpers (`triangulated_grid`, `tetrahedralized_grid`, `ball_mesh_2d`, `screen_to_world_2d`) that the tutorials import — there is nothing simulation-related in there, just mesh generation and screen-to-world ray math.

## Recurring concepts

A few ideas appear in almost every tutorial. Glance at these now and refer back as needed.

- **Rest pose `X` vs. current pose `U`.** `X` is the "undeformed" position of each vertex; `U` is where the vertex is right now. Strain is computed from the map `X → U`.
- **Mesh `T`.** A list of triangles (2D) or tetrahedra (3D). Each row holds the vertex indices for one element.
- **Deformation gradient `F`.** Per-element 2×2 (or 3×3) matrix that describes how a single triangle/tet has been stretched/rotated from rest. `F = I` means no deformation.
- **Energy `E(U)`.** A single scalar measuring how unhappy the material is in its current pose. Forces are `−∇E`. Simulating = repeatedly nudging `U` to reduce `E`.
- **Neo-Hookean energy.** The specific elastic energy used throughout (in [simkit/energies/](../../simkit/energies/)). It penalizes both stretching and volume change and blows up if a triangle inverts.
- **Soft pins (Dirichlet penalty).** Instead of hard-fixing a vertex to a target position, we add a stiff quadratic spring `½ K ‖x − target‖²` to the energy. The vertex is "pinned" because the spring is so stiff it barely moves. The pin contributes a constant matrix `Q_pin` and vector `b_pin` to the system — see [simkit/dirichlet_penalty.py](../../simkit/dirichlet_penalty.py).
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
- [001_deformation_gradient_demo.py:61](001_deformation_gradient_demo.py#L61) — `F = deformation_gradient(X, T, U).reshape((2, 2))` is the core mapping from rest+current vertex positions to a per-element matrix.
- [001_deformation_gradient_demo.py:69](001_deformation_gradient_demo.py#L69) — `energies.neo_hookean_energy_element_F(F, mu=1, lam=1)` evaluates the strain energy density given that `F`.
- Underlying math lives in [simkit/deformation_gradient.py](../../simkit/deformation_gradient.py) and [simkit/energies/neo_hookean.py](../../simkit/energies/neo_hookean.py).

---

## 002 — Picking and dragging a vertex (warmup)

[002_energy_demo.py](002_energy_demo.py)

Almost the same scene as 001 but reduced to just the interaction layer: right-click selects the nearest vertex, then holding `Space` while moving the mouse drags it. There is no energy readout — this file exists to keep the input handling minimal so the click/drag flow stays readable before more state gets layered on in the later demos.

**Focus code:**
- [002_energy_demo.py:23-38](002_energy_demo.py#L23-L38) — `screen_to_world_2d` shows how cursor pixel positions are converted into world-space coordinates by intersecting a camera ray with the `z = 0` plane. The same function is later imported from `utils.py` and used everywhere.

You can skip this file if you already understood the picking logic in 001.

---

## 003 — Resolution scaling: how mesh size affects solver cost

[003_resolution_complexity.py](003_resolution_complexity.py)

Three cantilever beams — coarse (24 verts), mid (100 verts), and fine (600 verts) — stacked vertically, all using the same physics (BDF2 implicit time integration, Newton solver). A radio button picks which one is currently running under gravity; the other two freeze. Each Newton iteration is wall-clock timed and the per-iteration cost is plotted live for each beam.

**What to learn:**
- Mesh refinement makes the dynamics more accurate but quadratically-to-cubically more expensive per step. The plot makes the cost gap visible — fine is roughly an order of magnitude slower than coarse.
- The bottleneck per Newton iteration is the sparse linear solve `H Δx = −g`, not the energy/gradient assembly.

**Focus code:**
- [003_resolution_complexity.py:108-128](003_resolution_complexity.py#L108-L128) — the explicit Newton loop. Each iteration assembles `g` and `H`, solves `H Δx = −g` with `scipy.sparse.linalg.spsolve`, line-searches an alpha, and updates. The `time.perf_counter()` calls around this body are what fill the timing plot.
- [003_resolution_complexity.py:100-106](003_resolution_complexity.py#L100-L106) — `Etot` shows the four terms that make up an implicit-dynamics energy: elastic + soft pin + gravity + kinetic.
- [003_resolution_complexity.py:77-78](003_resolution_complexity.py#L77-L78) — pieces of the Hessian that don't change between Newton iterations are cached outside the loop. Only the elastic Hessian is re-built every iteration.

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
- [004_solver_tradeoff.py:60-83](004_solver_tradeoff.py#L60-L83) — `total_energy`, `total_gradient`, `total_hessian`. Note that gradient descent only needs the first two; Newton needs all three.
- [004_solver_tradeoff.py:86-103](004_solver_tradeoff.py#L86-L103) — building either a [NewtonSolver](../../simkit/solvers/NewtonSolver.py) or [GradientDescentSolver](../../simkit/solvers/GradientDescentSolver.py) depending on the dropdown.
- [004_solver_tradeoff.py:150](004_solver_tradeoff.py#L150) — `solver.solve(...)` is the one-line API hiding the whole optimization loop.

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
- [005_integration_scheme.py:109-123](005_integration_scheme.py#L109-L123) — `implicit_step` builds `Etot = elastic + pin + gravity + kinetic` and hands it to a Newton solver. The kinetic energy is what encodes "we are doing dynamics, not statics."
- [005_integration_scheme.py:126-145](005_integration_scheme.py#L126-L145) — `step_be` and `step_bdf2` differ only in which `kinetic_energy_*` family they use: BE needs one previous pose, BDF2 needs three.
- [005_integration_scheme.py:148-163](005_integration_scheme.py#L148-L163) — `step_fe`, the explicit one. No Newton, no Hessian — just `x ← x + h v; v ← v + h M⁻¹ f`. Pinned DOFs are hard-clamped because Forward Euler cannot tolerate the stiff pin penalty.
- All three kinetic-energy variants live in [simkit/energies/](../../simkit/energies/) under `kinetic_energy_be`, `kinetic_energy_bdf2`, etc.

---

## 006 — Interactive static deformation

[006_interactive_deformation.py](006_interactive_deformation.py)

A beam pinned on the left. Left-click any other vertex and drag — that vertex becomes a soft-pinned handle that follows the cursor; release to let go. The solver minimizes elastic + pin + handle energy each frame, so the beam continuously catches up to wherever you're holding it. No mass, no gravity, no time — just static equilibrium.

This is the canonical "puppeteer a deformable" interaction loop. Compared to 004, the difference is that the handle vertex itself is now arbitrary (instead of a hardcoded vertex on the right edge), and the solver runs Newton each frame.

**Focus code:**
- [006_interactive_deformation.py:41-49](006_interactive_deformation.py#L41-L49) — `rebuild_handle()` rebuilds the `Q_handle`, `b_handle` penalty whenever the user grabs a new vertex or drags. This is just `dirichlet_penalty` applied to a single index.
- [006_interactive_deformation.py:57-75](006_interactive_deformation.py#L57-L75) — total energy/gradient/Hessian compose two pin penalties on top of the elastic term.
- [006_interactive_deformation.py:137](006_interactive_deformation.py#L137) — `solver.solve(...)` is called once per frame; the implicit assumption is that the handle moves slowly relative to the solver's convergence rate.

---

## 007 — Interactive dynamics

[007_interactive_dynamics.py](007_interactive_dynamics.py)

Same beam, same handle UI as 006, but now with mass: a kinetic-energy term is added, the integrator is selectable (BE / BDF2 / FE), and `dt` is a slider. Grabbing a vertex while the beam is mid-swing throws inertia around — you can see the beam overshoot, ring, and settle.

This is the merge of 005 (integrators) and 006 (handle UI). Use it to feel the difference between implicit damping (BE settles fast) and energy-preserving behavior (BDF2 keeps ringing).

**Focus code:**
- [007_interactive_dynamics.py:100-138](007_interactive_dynamics.py#L100-L138) — same `implicit_step` / `step_be` / `step_bdf2` / `step_fe` structure as in tutorial 005; the only addition is the handle penalty inside `pin_E`/`pin_g`/`pin_H`.
- [007_interactive_dynamics.py:151-165](007_interactive_dynamics.py#L151-L165) — `advance()` reads the current integrator choice and rotates the BDF2 history buffers (`U_prev`, `U_prev2`, `U_prev3`).

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
- [008_interactive_contact_ball.py:124-142](008_interactive_contact_ball.py#L124-L142) — `contact_E`, `contact_g`, `contact_H` call into [simkit/energies/](../../simkit/energies/) `contact_springs_sphere_*`. The contact energy is the per-vertex sum of `½ K · M · max(0, R − ‖x − ball_p‖)²` — i.e., a one-sided spring that only activates when a vertex is inside the ball.
- [008_interactive_contact_ball.py:150-170](008_interactive_contact_ball.py#L150-L170) — `implicit_step` now sums **four** potentials: elastic + pin + contact + gravity, plus the kinetic term for whichever integrator is selected.
- [008_interactive_contact_ball.py:38-43](008_interactive_contact_ball.py#L38-L43) — `lame_from_E_nu` converts the more intuitive `(E, ν)` material parameters into the Lamé parameters `(μ, λ)` that the neo-Hookean energy actually consumes.

---

## 009 — Contact against a static floor (2D)

[009_interactive_contact_plane.py](009_interactive_contact_plane.py)

A patch sits above a horizontal floor (a line at `y = floor_y`). Gravity pulls it down; the floor pushes it up via plane-contact penalty springs. Left-click + drag any vertex to throw the patch around, then release to let it fall back.

**What to try:**
- Just let it sit — it should rest on the floor with a small visible bulge where vertices are slightly below the floor (penalty methods always allow some penetration).
- Grab a corner and lift the patch high, then drop it — watch it bounce and settle.
- Switch to Forward Euler and slowly increase `dt` — the contact spring becomes the stiffest mode and is the first thing to explode.

**Focus code:**
- [009_interactive_contact_plane.py:117-126](009_interactive_contact_plane.py#L117-L126) — `contact_*` now uses `contact_springs_plane_*` instead of the sphere version. The math: for each vertex, `signed_distance = (x − floor_p) · floor_n`, and the energy is `½ K · M · max(0, −signed_distance)²` — same one-sided-spring idea as the ball, but against a half-space.
- [009_interactive_contact_plane.py:144-164](009_interactive_contact_plane.py#L144-L164) — the implicit step sums elastic + contact + handle + gravity + kinetic. Structure is the same as the previous tutorials; only the contact term changed.

---

## 010 — Contact in 3D

[010_interactive_contact_plane_3D.py](010_interactive_contact_plane_3D.py)

The 2D scene from 009 lifted into 3D: a tetrahedral block dropped onto a floor. Same neo-Hookean elasticity, same plane-penalty contact, same integrators. The mesh is built from a hex grid that splits each cell into five tets (see `tetrahedralized_grid` in [utils.py](utils.py)).

**What to try / how to interact:**
- The mouse owns two modes. By default, the "Click-to-drag handle" checkbox is **on** — left-click a vertex and drag to deform the block in 3D. To orbit/zoom the camera, **uncheck** the box (polyscope takes the mouse back).
- Picking in 3D uses polyscope's `screen_coords_to_world_position` to find the world point under the cursor, then snaps to the nearest mesh vertex. Dragging keeps the handle on a camera-facing plane through the picked point — see [010_interactive_contact_plane_3D.py:230-267](010_interactive_contact_plane_3D.py#L230-L267).
- Crank Young's modulus high — the block barely deforms on impact. Lower it — the block squishes flat and slowly recovers.

**Focus code:**
- Almost everything is identical to 009; the only real changes are (a) `dim = 3` everywhere automatically because `X` is now `(n, 3)`, (b) the mesh is a volume mesh (`ps.register_volume_mesh`), and (c) the picking logic uses a real 3D ray instead of a `z = 0` plane intersection.
- [010_interactive_contact_plane_3D.py:235-244](010_interactive_contact_plane_3D.py#L235-L244) — `pick_world_point`: where the cursor is in world space.
- [010_interactive_contact_plane_3D.py:261-267](010_interactive_contact_plane_3D.py#L261-L267) — `intersect_ray_plane`: how dragging stays consistent across the screen.

---

## Where to go next

The tutorials cover the spine of an FEM simulator: deformation gradient → energy → static solve → dynamic solve → contact → 3D. Once these feel comfortable, look at:

- [simkit/energies/](../../simkit/energies/) — the actual neo-Hookean / kinetic / contact energy implementations, including their analytic gradients and PSD-projected Hessians.
- [simkit/solvers/](../../simkit/solvers/) — `NewtonSolver`, `GradientDescentSolver`, and the line-search and backtracking utilities.
- [examples/](..) — larger end-to-end scenes that compose these same primitives.
