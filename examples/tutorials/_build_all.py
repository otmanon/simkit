"""Generator for the full offline SimKit notebook tutorial series (01-10).

    python _build_all.py

Design notes
------------
* Clean header: ``import simkit`` + ``import simkit.energies as energies`` +
  (for the solver tutorials) the NewtonSolver import. No long per-function lists.
* simkit's ``*_energy_x`` / contact / kinetic energies already return Python
  ``float``, so there are no ``float(...)`` wrappers in the notebooks.
* Tutorial 01 builds the deformation gradient from scratch; everything else is
  shifted up by one from the previous series.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PATH = ('# --- make the *local* simkit (this repo) importable, ahead of any installed copy ---\n'
        'import sys, os\n'
        'sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))\n'
        '%matplotlib inline')

BOOT_LIGHT = PATH + '''
import numpy as np
import matplotlib.pyplot as plt
import simkit
import simkit.energies as energies
import utils
os.makedirs("media", exist_ok=True)'''

BOOT_SIM = PATH + '''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import simkit
import simkit.energies as energies
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
import utils
os.makedirs("media", exist_ok=True)'''


def md(s): return new_markdown_cell(s)
def code(s): return new_code_cell(s)


def save(cells, path):
    nb = new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


# =============================================================================
# 01 - Building the Deformation Gradient
# =============================================================================
def t01():
    c = []
    c.append(md(
"""# 1 &middot; Building the Deformation Gradient

Every tutorial after this one leans on one object: the **deformation gradient**
\\(F\\). Before we *use* it, let's *build* it, so it never feels like it appears
by magic.

A deformation is a map \\(\\phi\\) that takes each material (rest) point
\\(X\\) to a deformed point \\(x = \\phi(X)\\). The deformation gradient is its
Jacobian,
$$ F = \\frac{\\partial x}{\\partial X}, $$
a local linear map that says how a tiny chunk of material around \\(X\\) is
stretched and rotated. For a single triangle the map is **affine**,
\\(x = F X + t\\), so \\(F\\) is one constant \\(2\\times 2\\) matrix we can solve
for directly."""))
    c.append(code(BOOT_LIGHT))
    c.append(md(
"""## Eliminate the translation with edge vectors

If \\(x = F X + t\\), then subtracting vertex 0 from vertices 1 and 2 cancels the
unknown translation \\(t\\):
$$ x_i - x_0 = F\\,(X_i - X_0). $$
Stack the two rest edges as columns of \\(D_m = [\\,X_1-X_0,\\; X_2-X_0\\,]\\) and the
two deformed edges as \\(D_s = [\\,x_1-x_0,\\; x_2-x_0\\,]\\). Then \\(D_s = F D_m\\),
so
$$ \\boxed{\\,F = D_s\\, D_m^{-1}\\,}. $$
That's the whole thing &mdash; two edge matrices and one inverse."""))
    c.append(code(
'''def deformation_gradient_triangle(X, U):
    """F for a single triangle, built from rest edges (X) and deformed edges (U)."""
    Dm = np.column_stack([X[1] - X[0], X[2] - X[0]])   # rest edge vectors
    Ds = np.column_stack([U[1] - U[0], U[2] - U[0]])   # deformed edge vectors
    return Ds @ np.linalg.inv(Dm)

# rest triangle (equilateral, side 1)
X = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, np.sqrt(3) / 2]])

# sanity checks ---------------------------------------------------------------
print("rest -> rest gives the identity:\\n", np.round(deformation_gradient_triangle(X, X), 6))

theta = np.deg2rad(30)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
U_rot = (X @ R.T) + np.array([2.0, -1.0])          # rotate AND translate
print("\\nrotate 30 deg + translate -> F is exactly that rotation:\\n",
      np.round(deformation_gradient_triangle(X, U_rot), 6))'''))
    c.append(md(
"""Two things already stand out:

* **Translation drops out.** We built \\(F\\) from *edges*, and translating every
  vertex leaves the edges unchanged. So \\(F\\) is blind to where the triangle
  sits &mdash; only its shape matters.
* A **rotation** maps to a rotation matrix \\(F = R\\) (and \\(\\det F = 1\\): no area
  change)."""))
    c.append(md(
"""## The same thing, for any simplex: `simkit.deformation_gradient`

SimKit generalizes this to triangles *and* tetrahedra with a reference-element
shape-function gradient (the matrix `H`), but it computes exactly the same
\\(F\\). Let's confirm our hand-built version matches it."""))
    c.append(code(
'''T = np.array([[0, 1, 2]])                  # one triangle, vertices 0,1,2

ours    = deformation_gradient_triangle(X, U_rot)
simkit_F = simkit.deformation_gradient(X, T, U_rot)[0]    # [0] = first (only) element

print("match:", np.allclose(ours, simkit_F))
print(np.round(simkit_F, 6))'''))
    c.append(md(
"""## Seeing it

The rest triangle (dashed) and three deformed versions, each labelled with the
\\(F\\) we just built. Note the rotate-*and*-translate case: a big translation, yet
\\(F\\) is still a pure rotation with \\(\\det F = 1\\)."""))
    c.append(code(
'''cases = [("rotate + translate", U_rot),
         ("scale x1.6, y0.7",   X * np.array([1.6, 0.7])),
         ("shear k = 0.7",      X @ np.array([[1.0, 0.7], [0.0, 1.0]]).T)]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (name, U) in zip(axes, cases):
    utils.setup_axes(ax, (-1.5, 2.8), (-1.4, 1.6), title=name)
    utils.TriangleArtist(ax, U, rest=X)
    utils.text_box(ax, utils.format_F(deformation_gradient_triangle(X, U)), loc="lower right")
fig.suptitle(r"$F = D_s D_m^{-1}$  built by hand, for three deformations")
fig.tight_layout(); plt.show()'''))
    c.append(md(
"""### Takeaways
* \\(F = D_s D_m^{-1}\\): the deformed edges expressed in the rest-edge basis.
* It is the **local linear part** of the deformation &mdash; translation-free by
  construction.
* `simkit.deformation_gradient(X, T, U)` returns one \\(F\\) per element and is
  what every later tutorial calls.

Next we build intuition for what different \\(F\\)'s *look like*."""))
    save(c, "01_deformation_gradient_construction.ipynb")


# =============================================================================
# 02 - Deformation Gradient Intuition
# =============================================================================
def t02():
    c = []
    c.append(md(
"""# 2 &middot; Deformation Gradient Intuition

Now that we know \\(F = D_s D_m^{-1}\\) (tutorial 1), let's *feel* it. We apply the
four basic deformations &mdash; **translation, rotation, scale, shear** &mdash; to
a triangle and read off \\(F\\) each time.

Watch for the headline fact: **\\(F\\) ignores translation.** A gradient kills
additive constants, so sliding the triangle around leaves \\(F = I\\)."""))
    c.append(code(BOOT_LIGHT))
    c.append(md(
"""## The rest triangle and four deformation maps

Each map is a linear transform of the rest vertices (translation also adds an
offset). We read \\(F\\) back with `simkit.deformation_gradient`."""))
    c.append(code(
'''X = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, np.sqrt(3) / 2]])
T = np.array([[0, 1, 2]])

def translate(X, t):     return X + np.asarray(t, float)
def rotate(X, theta):
    c, s = np.cos(theta), np.sin(theta)
    return X @ np.array([[c, -s], [s, c]]).T
def scale(X, sx, sy=None):
    sy = sx if sy is None else sy
    return X * np.array([sx, sy])
def shear(X, k):         return X @ np.array([[1.0, k], [0.0, 1.0]]).T

def F_of(U):             return simkit.deformation_gradient(X, T, U)[0]'''))
    c.append(md(
"""## Translation leaves \\(F = I\\)

Slide the triangle anywhere; \\(F\\) stays the identity."""))
    c.append(code(
'''for offset in ([0.0, 0.0], [2.0, -1.5], [-3.0, 4.0]):
    F = F_of(translate(X, offset))
    print(f"translate by {offset}:  F =\\n{np.round(F, 3)}   det F = {np.linalg.det(F):.3f}\\n")'''))
    c.append(md(
"""## The four deformations side by side

Rotation gives a pure rotation (\\(\\det F = 1\\)); scale gives a diagonal \\(F\\);
shear gives an off-diagonal term; translation &mdash; even with a big offset
&mdash; gives \\(F = I\\)."""))
    c.append(code(
'''cases = [
    ("translate (+offset)", translate(X, [1.8, 0.6])),
    ("rotate 45 deg",        rotate(X, np.pi / 4)),
    ("scale x1.6, y0.7",     scale(X, 1.6, 0.7)),
    ("shear k = 0.8",        shear(X, 0.8)),
]
cases = [(name, U, F_of(U)) for name, U in cases]
fig, _ = utils.deformation_panels(cases, rest=X)
plt.show()'''))
    c.append(md("""## Animation: rotating the triangle &mdash; \\(F\\) traces \\(R(\\theta)\\)"""))
    c.append(code(
'''angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
states = [rotate(X, a) for a in angles]
Fs = [F_of(U) for U in states]
fig, anim = utils.animate_deformation(states, Fs, rest=X, fps=20,
                                      title="Pure rotation:  F = R(theta)")
utils.save_anim(anim, "media/02_rotation.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md("""## Animation: translating the triangle &mdash; \\(F = I\\) always"""))
    c.append(code(
'''ts = np.linspace(0, 2 * np.pi, 48)
path = np.stack([1.6 * np.sin(ts), 1.0 * np.sin(2 * ts)], axis=1)   # figure-8
states = [translate(X, p) for p in path]
Fs = [F_of(U) for U in states]
fig, anim = utils.animate_deformation(states, Fs, rest=X,
                                      lims=((-2.6, 2.6), (-2.0, 2.6)), fps=20,
                                      title="Pure translation:  F = I  (always)")
utils.save_anim(anim, "media/02_translation.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""### Takeaways
* \\(F\\) is the local gradient, so it is **blind to translation**.
* Rotation &rarr; rotation matrix; scale &amp; shear show up directly in \\(F\\).

Next: turn \\(F\\) into a single number measuring *how deformed* a shape is &mdash;
the **elastic energy**."""))
    save(c, "02_deformation_gradient_intuition.ipynb")


# =============================================================================
# 03 - Elastic Energy (energies defined inline)
# =============================================================================
def t03():
    c = []
    c.append(md(
"""# 3 &middot; Elastic Energy Intuition

The deformation gradient \\(F\\) says *how* a triangle is deformed. An **elastic
energy** \\(\\psi(F)\\) turns that into a single number: *how much* it costs.

We will **write each energy out ourselves** &mdash; one line of math each &mdash;
so nothing is hidden, then compare three classics:

* **Linear elasticity**
* **ARAP** (as-rigid-as-possible)
* **Neo-Hookean** (classical)

A good energy is **zero for rigid motions** (translation, rotation) and grows
with genuine deformation (scale, shear). We'll see linear elasticity fail the
rotation test, and Neo-Hookean blow up as a triangle is crushed flat."""))
    c.append(code(BOOT_LIGHT))
    c.append(md(
"""## The three energies, written out

All three are functions of \\(F\\) alone, with material constants \\(\\mu\\) (shear)
and \\(\\lambda\\) (bulk). Let \\(I\\) be the identity and \\(\\dim = 2\\) here.

**Linear elasticity** &mdash; small-strain tensor \\(\\varepsilon = \\tfrac12(F + F^\\top) - I\\):
$$ \\psi = \\mu\\,\\lVert\\varepsilon\\rVert_F^2 + \\tfrac{\\lambda}{2}\\,\\mathrm{tr}(\\varepsilon)^2. $$

**ARAP** &mdash; \\(R\\) is the rotation from the polar decomposition of \\(F\\); it
measures the distance from the *nearest pure rotation*:
$$ \\psi = \\mu\\,\\lVert F - R\\rVert_F^2. $$

**Neo-Hookean** &mdash; with \\(I_C = \\lVert F\\rVert_F^2\\) and \\(J = \\det F\\) (the
area ratio). The \\(-\\mu\\ln J\\) term is the key one: it &rarr; \\(+\\infty\\) as
\\(J \\to 0\\), forbidding the material from being crushed to zero area:
$$ \\psi = \\tfrac{\\mu}{2}(I_C - 2)\\; -\\; \\mu\\ln J\\; +\\; \\tfrac{\\lambda}{2}(\\ln J)^2. $$"""))
    c.append(code(
'''def linear_energy(F, mu=1.0, lam=1.0):
    eps = 0.5 * (F + F.T) - np.eye(2)                    # small-strain tensor
    E_shear  = mu * np.sum(eps**2)                       # deviatoric (shape) term
    E_volume = 0.5 * lam * np.trace(eps)**2             # volumetric term
    E_total  = E_shear + E_volume
    return E_total

def arap_energy(F, mu=1.0, lam=1.0):
    R = simkit.polar_svd(F[None])[0][0]                  # nearest rotation to F
    return mu * np.sum((F - R)**2)                       # single term; ARAP ignores lam

def neo_hookean_energy(F, mu=1.0, lam=1.0):
    I_C = np.sum(F**2)
    J = np.linalg.det(F)
    E_shear   = 0.5 * mu * (I_C - 2)                     # resists shearing/stretching
    E_barrier = -mu * np.log(J)                          # forbids inversion (J -> 0)
    E_volume  = 0.5 * lam * np.log(J)**2                # resists volume change
    E_total   = E_shear + E_barrier + E_volume
    return E_total

ENERGIES = {"Linear": linear_energy, "ARAP": arap_energy, "Neo-Hookean": neo_hookean_energy}

# deformation maps and a helper that evaluates all three energies over a sweep
X = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, np.sqrt(3) / 2]])
T = np.array([[0, 1, 2]])
def translate(X, t): return X + np.asarray(t, float)
def rotate(X, th):
    c, s = np.cos(th), np.sin(th); return X @ np.array([[c, -s], [s, c]]).T
def scale(X, s):     return X * np.array([s, s])
def shear(X, k):     return X @ np.array([[1.0, k], [0.0, 1.0]]).T

def sweep(states, mu=1.0, lam=1.0):
    """Energy of each model across a list of deformed triangles."""
    Fs = [simkit.deformation_gradient(X, T, U)[0] for U in states]
    return {name: np.array([fn(F, mu, lam) for F in Fs]) for name, fn in ENERGIES.items()}'''))
    c.append(md(
"""## Translation costs no energy

Sliding the triangle leaves every energy flat at zero. (We fix the y-axis so the
flat line doesn't look like noise.)"""))
    c.append(code(
'''d = np.linspace(0, 3, 40)
states = [translate(X, [dx, 0.0]) for dx in d]
series = sweep(states)
utils.line_plot(d, series, xlabel="translation distance", ylabel="elastic energy",
                title="Translation costs no energy", ylim=(-0.5, 2.0))
plt.show()

fig, anim = utils.animate_scene_energy(states, d, series, rest=X,
    xlabel="translation distance", lims=((-1.2, 4.0), (-1.2, 1.6)),
    scene_title="triangle sliding", title="energy vs translation", fps=20,
    energy_ylim=(-0.5, 2.0))      # fixed y-axis (matches the static plot) -> flat line
utils.save_anim(anim, "media/03_translation.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## Scale: every energy grows (the intuitive case)

Uniform scaling genuinely deforms the material, so all three energies rise as we
move away from \\(s = 1\\)."""))
    c.append(code(
'''s = np.linspace(0.5, 1.8, 50)
states = [scale(X, si) for si in s]
series = sweep(states)
utils.line_plot(s, series, xlabel="scale factor s", ylabel="elastic energy",
                title="Energy vs uniform scale")
plt.show()
fig, anim = utils.animate_scene_energy(states, s, series, rest=X, xlabel="scale factor s",
    scene_title="triangle scaling", title="energy vs scale", fps=20)
utils.save_anim(anim, "media/03_scale.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md("""## Shear: every energy grows too""")  )
    c.append(code(
'''k = np.linspace(0.0, 1.6, 50)
states = [shear(X, ki) for ki in k]
series = sweep(states)
utils.line_plot(k, series, xlabel="shear k", ylabel="elastic energy",
                title="Energy vs shear")
plt.show()
fig, anim = utils.animate_scene_energy(states, k, series, rest=X, xlabel="shear k",
    scene_title="triangle shearing", title="energy vs shear", fps=20)
utils.save_anim(anim, "media/03_shear.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## Rotation: linear elasticity is *not* rotation-invariant

Now the surprise. Spin the triangle through a full turn. **ARAP** and
**Neo-Hookean** stay at zero &mdash; rotation is free. **Linear elasticity**
rises and falls: its small-strain tensor \\(\\varepsilon\\) mistakes rotation for
stretch. This is why linear elasticity is only valid for *tiny* deformations."""))
    c.append(code(
'''ang = np.linspace(0, 2 * np.pi, 60)
states = [rotate(X, a) for a in ang]
series = sweep(states)
utils.line_plot(ang, series, xlabel="rotation angle (rad)", ylabel="elastic energy",
                title="Only linear elasticity penalizes rotation")
plt.show()
fig, anim = utils.animate_scene_energy(states, ang, series, rest=X,
    xlabel="rotation angle (rad)", scene_title="triangle rotating",
    title="energy vs rotation angle", fps=20)
utils.save_anim(anim, "media/03_rotation.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## Crushing the triangle flat: Neo-Hookean &rarr; \\(\\infty\\)

Drive the apex down toward the base so the area \\(J = \\det F \\to 0\\).

**Why didn't it explode before?** The Neo-Hookean blow-up is real but only
*logarithmic*: \\(\\psi \\approx \\tfrac{\\lambda}{2}(\\ln J)^2 - \\mu\\ln J\\). Since
\\(\\ln J\\) grows slowly, you have to (a) drive \\(J\\) *extremely* close to zero
**and** (b) use a stiff, nearly-incompressible material (large \\(\\lambda\\)) for the
number to actually rocket. So here we collapse hard &mdash; \\(y = y_0\\,10^{-9t}\\),
i.e. \\(J\\) down to \\(\\sim 10^{-9}\\) &mdash; with \\(\\lambda = 50\\). Now
**Neo-Hookean reaches \\(\\sim 10^4\\)** and is climbing to \\(+\\infty\\), while
**ARAP** sits at a finite plateau (it has no volume term) and **linear** stays
modest. We plot on a log axis."""))
    c.append(code(
'''t = np.linspace(0, 1, 50)
y0 = X[2, 1]
def collapse_apex(frac):
    U = X.copy(); U[2, 1] = y0 * 10.0**(-9.0 * frac); return U   # hard log collapse -> J ~ 1e-9
states = [collapse_apex(f) for f in t]
series = sweep(states, mu=1.0, lam=50.0)              # nearly-incompressible: lam >> mu
Jend = np.linalg.det(simkit.deformation_gradient(X, T, states[-1])[0])
print(f"det F at full collapse: {Jend:.2e}   Neo-Hookean energy: {series['Neo-Hookean'][-1]:.0f}")

utils.line_plot(t, series, xlabel="collapse parameter", ylabel="elastic energy",
                logy=True, title="Neo-Hookean rockets to infinity as area -> 0")
plt.show()
fig, anim = utils.animate_scene_energy(states, t, series, rest=X,
    xlabel="collapse parameter", scene_title="apex crushed onto the base",
    title="energy vs collapse", logy=True, fps=18)
utils.save_anim(anim, "media/03_collapse.mp4", fps=18)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""### Takeaways
* We wrote each energy in one line: linear (strain tensor), ARAP (distance to a
  rotation), Neo-Hookean (with the \\(-\\mu\\ln J\\) inversion barrier).
* Good energies are zero for rigid motion and grow with scale/shear.
* **Linear** breaks rotation-invariance; **Neo-Hookean** diverges at collapse;
  **ARAP** saturates.

Next: minimize an energy (plus constraints) to actually *pose* a shape."""))
    save(c, "03_elastic_energy.ipynb")


# =============================================================================
# 04 - Elastostatics as a minimization problem
# =============================================================================
def t04():
    c = []
    c.append(md(
"""# 4 &middot; Elastostatics as a Minimization Problem

Posing an elastic object is an **optimization**: find the vertex positions
\\(x\\) that minimize
$$ E(x) = \\underbrace{E_{\\text{elastic}}(x)}_{\\text{wants rest shape}}
        + \\underbrace{\\tfrac{K}{2}\\lVert x_{\\text{pin}} - p\\rVert^2}_{\\text{hold pins}}
        + \\underbrace{\\tfrac{K}{2}\\lVert x_{\\text{handle}} - h\\rVert^2}_{\\text{drag handle}}. $$

We pin part of the shape, drag a **handle**, and let everything else settle. The
pin/handle terms are quadratic *spring* penalties from
`simkit.dirichlet_penalty`. We minimize silently with Newton (the solver is
tutorial 5's job) and focus on how **different energies pose differently**."""))
    c.append(code(BOOT_SIM))
    c.append(md(
"""## A small, readable objective

`make_material` picks the elastic model (its energy / gradient / PSD-projected
Hessian, all with the same signature). `ElasticPose` adds two spring terms
&mdash; a fixed **pin** and a movable **handle** &mdash; and exposes
`energy / gradient / hessian` for the Newton solver."""))
    c.append(code(
'''def make_material(name):
    """(energy, gradient, hessian) of the elastic term, uniform (xn, J, mu, lam, vol) signature."""
    E = energies
    if name == "ARAP":          # ARAP uses only mu
        return (lambda xn, J, mu, lam, vol: E.arap_energy_x(xn, J, mu, vol),
                lambda xn, J, mu, lam, vol: E.arap_gradient_x(xn, J, mu, vol),
                lambda xn, J, mu, lam, vol: E.arap_hessian_x(xn, J, mu, vol, psd=True))
    if name == "Linear":
        return (E.linear_elasticity_energy_x, E.linear_elasticity_gradient_x,
                lambda xn, J, mu, lam, vol: E.linear_elasticity_hessian_x(xn, J, mu, lam, vol, psd=True))
    return (E.macklin_mueller_neo_hookean_energy_x, E.macklin_mueller_neo_hookean_gradient_x,
            lambda xn, J, mu, lam, vol: E.macklin_mueller_neo_hookean_hessian_x(xn, J, mu, lam, vol, psd=True))


class ElasticPose:
    """Minimize  elastic(x) + pin spring + handle spring."""

    def __init__(self, X, T, material="Neo-Hookean", K=1e5):
        self.X, self.T, self.K = X, T, K
        self.n, self.dim = X.shape
        self.J, self.vol = simkit.deformation_jacobian(X, T), simkit.volume(X, T)
        self.mu  = np.full((len(T), 1), 1.0)
        self.lam = np.full((len(T), 1), 1.0)
        self.elastic_energy, self.elastic_grad, self.elastic_hess = make_material(material)
        self.pin = self._spring([], np.empty((0, self.dim)))      # (Q, b)
        self.handle = self._spring([], np.empty((0, self.dim)))

    def _spring(self, idx, targets):
        return simkit.dirichlet_penalty(np.atleast_1d(np.asarray(idx, int)),
                                        np.atleast_2d(targets), self.n, self.K)

    def set_pin(self, idx, targets):     self.pin = self._spring(idx, targets)
    def set_handle(self, idx, targets):  self.handle = self._spring(idx, targets)

    # spring penalty 1/2 x^T Q x + b^T x, and its gradient / Hessian
    @staticmethod
    def _spring_energy(Qb, xc):  Q, b = Qb; return 0.5 * (xc.T @ (Q @ xc))[0, 0] + (b.T @ xc)[0, 0]
    @staticmethod
    def _spring_grad(Qb, xc):    Q, b = Qb; return Q @ xc + b

    def energy(self, x):
        xn, xc = x.reshape(-1, self.dim), x.reshape(-1, 1)
        E_elastic = self.elastic_energy(xn, self.J, self.mu, self.lam, self.vol)
        E_pin     = self._spring_energy(self.pin, xc)
        E_handle  = self._spring_energy(self.handle, xc)
        E_total   = E_elastic + E_pin + E_handle
        return E_total

    def gradient(self, x):
        xn, xc = x.reshape(-1, self.dim), x.reshape(-1, 1)
        g_elastic = self.elastic_grad(xn, self.J, self.mu, self.lam, self.vol)
        g_pin     = self._spring_grad(self.pin, xc)
        g_handle  = self._spring_grad(self.handle, xc)
        g_total   = g_elastic + g_pin + g_handle
        return g_total

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_elastic = self.elastic_hess(xn, self.J, self.mu, self.lam, self.vol)
        H_pin     = self.pin[0]                          # Q of the pin spring
        H_handle  = self.handle[0]                       # Q of the handle spring
        H_total   = H_elastic + H_pin + H_handle
        return H_total

    def solve(self, U0, iters=30):
        x = NewtonSolver(self.energy, self.gradient, self.hessian,
                         NewtonSolverParams(max_iter=iters, do_line_search=True)
                         ).solve(U0.reshape(-1, 1))
        return x.reshape(self.n, self.dim)'''))
    c.append(md(
"""## A triangle: pin the left vertex, swing the right one in a figure-8

The pinned vertex stays put, the handle traces a **figure-8** (so we see the
free apex both stretch and compress), and the apex follows differently for each
elastic model."""))
    c.append(code(
'''X = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, np.sqrt(3) / 2]])
T = np.array([[0, 1, 2]])
PIN, HANDLE = 0, 1

# figure-8 (lemniscate) traced by the handle, starting from rest (t=0 -> no offset)
t = np.linspace(0, 2 * np.pi, 48)
fig8 = np.stack([0.45 * np.sin(t), 0.6 * np.sin(2 * t)], axis=1)   # half as wide as before
handle_path = X[HANDLE] + fig8

def run(material):
    pose = ElasticPose(X, T, material)
    pose.set_pin(PIN, X[PIN])
    states, U = [], X.copy()
    for h in handle_path:
        pose.set_handle(HANDLE, h)
        U = pose.solve(U)
        states.append(U.copy())
    return states

states = run("Neo-Hookean")
fig, anim = utils.animate_mesh(states, T, lims=((-1.2, 1.6), (-1.2, 1.2)),
    title="Neo-Hookean triangle: handle traces a figure-8",
    pin_pts=X[[PIN]], handle_traj=[s[HANDLE] for s in states],
    target_pts=list(handle_path), fps=20, figsize=(6, 5))
utils.save_anim(anim, "media/04_triangle.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## Same handle, three energies, very different poses

Pull the handle far up-and-right and compare where the free apex lands under each
model. Linear elasticity (no rotation-invariance) reacts quite differently from
ARAP and Neo-Hookean."""))
    c.append(code(
'''target = np.array([1.5, 1.1])
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, material in zip(axes, ["Linear", "ARAP", "Neo-Hookean"]):
    pose = ElasticPose(X, T, material)
    pose.set_pin(PIN, X[PIN]); pose.set_handle(HANDLE, target)
    U = pose.solve(X.copy())
    utils.setup_axes(ax, (-1.0, 2.0), (-0.8, 1.6), title=material)
    utils.TriangleArtist(ax, U, rest=X)
    ax.plot(*X[PIN], "s", color=utils.PIN_C, ms=10)
    ax.plot(*target, "x", color="0.3", ms=12, mew=2.5)
fig.suptitle("Same pin + handle target, three elastic energies")
fig.tight_layout(); plt.show()'''))
    c.append(md(
"""## A cantilever beam: pin the left edge, swing the whole right edge

The handle is now **every vertex on the right edge** (not one). We **rotate those
target points about the origin** &mdash; the orange handles swing up and down in a
wide arc &mdash; which whips the beam through large bends. Starting from rest, the
angle sweeps to about \\(\\pm 120^\\circ\\) and back. Targets are marked with an
**&times;**."""))
    c.append(code(
'''Xb, Tb = utils.triangulated_grid(nx=16, ny=5, width=2.0, height=0.4)
pin_idx   = np.where(Xb[:, 0] <= Xb[:, 0].min() + 1e-6)[0]
right_idx = np.where(Xb[:, 0] >= Xb[:, 0].max() - 1e-6)[0]

# rotate the right-edge target points about the ORIGIN; angle swings +/-120 deg from rest
angles = np.deg2rad(120) * np.sin(np.linspace(0, 2 * np.pi, 48))
def rotate_about_origin(P, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return P @ R.T

beam = ElasticPose(Xb, Tb, "Neo-Hookean")
beam.set_pin(pin_idx, Xb[pin_idx])
states, targets, U = [], [], Xb.copy()
for th in angles:
    tgt = rotate_about_origin(Xb[right_idx], th)
    beam.set_handle(right_idx, tgt)
    U = beam.solve(U)
    states.append(U.copy()); targets.append(tgt.copy())

fig, anim = utils.animate_mesh(states, Tb, lims=((-1.8, 1.8), (-1.8, 1.8)),
    title="Cantilever: right edge swung in an arc about the origin",
    pin_pts=Xb[pin_idx],
    handle_traj=[s[right_idx] for s in states], target_pts=targets, fps=20,
    figsize=(6, 6))
utils.save_anim(anim, "media/04_beam.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""### Takeaways
* Posing = **minimizing** elastic energy + soft pin/handle springs.
* The *same* setup with a *different* energy gives a *different* equilibrium.
* We solved it silently with Newton. **How** that solve works is next."""))
    save(c, "04_elastostatics_minimization.ipynb")


# =============================================================================
# 05 - Numerical solver: Newton vs Gradient Descent
# =============================================================================
def t05():
    c = []
    c.append(md(
"""# 5 &middot; Solving the Minimization: Newton vs. Gradient Descent

Time to open the solver box. We pin a beam's left edge, grab its **whole right
edge** as a handle, fix a target down-and-to-the-right, and drive the beam to
equilibrium from rest with two methods:

* **Gradient descent** &mdash; step along \\(-\\nabla E\\).
* **Newton** &mdash; step along \\(-H^{-1}\\nabla E\\), using curvature.

Per iteration we track the energy, the gradient norm \\(\\lVert\\nabla E\\rVert\\),
and the **Newton decrement** \\(\\lambda = \\sqrt{\\nabla E^\\top H^{-1}\\nabla E}\\)."""))
    c.append(code(BOOT_SIM))
    c.append(md(
"""## The objective

Elastic + pinned-left-edge + right-edge handle, all stable Neo-Hookean. We keep
the **handle spring soft** (similar magnitude to the elastic forces) so the
problem is not horribly ill-conditioned &mdash; otherwise gradient descent just
yanks the handle and leaves the rest of the beam untouched. The
`energy / gradient / hessian` read like the math; `quad` is the spring penalty
\\(\\tfrac12 x^\\top Q x + b^\\top x\\)."""))
    c.append(code(
'''X, T = utils.triangulated_grid(nx=12, ny=4, width=2.0, height=0.4)
n, dim = X.shape
J, vol = simkit.deformation_jacobian(X, T), simkit.volume(X, T)
mu, lam = simkit.ympr_to_lame(0.5, 0.45)                       # soft beam
mu  = np.full((len(T), 1), mu)
lam = np.full((len(T), 1), lam)

pin_idx   = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
right_idx = np.where(X[:, 0] >= X[:, 0].max() - 1e-6)[0]
Q_pin, b_pin = simkit.dirichlet_penalty(pin_idx, X[pin_idx], n, 1e2)   # gentle pin
targets = X[right_idx] + np.array([0.5, -0.7])                 # pull down-and-right
Q_h, b_h = simkit.dirichlet_penalty(right_idx, targets, n, 25.0)       # soft handle

def quad(Q, b, xc):  return 0.5 * (xc.T @ (Q @ xc))[0, 0] + (b.T @ xc)[0, 0]

def energy(x):
    xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
    E_elastic = energies.macklin_mueller_neo_hookean_energy_x(xn, J, mu, lam, vol)
    E_pin     = quad(Q_pin, b_pin, xc)
    E_handle  = quad(Q_h, b_h, xc)
    E_total   = E_elastic + E_pin + E_handle
    return E_total
def gradient(x):
    xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
    g_elastic = energies.macklin_mueller_neo_hookean_gradient_x(xn, J, mu, lam, vol)
    g_pin     = Q_pin @ xc + b_pin
    g_handle  = Q_h @ xc + b_h
    g_total   = g_elastic + g_pin + g_handle
    return g_total
def hessian(x):
    xn = x.reshape(-1, dim)
    H_elastic = energies.macklin_mueller_neo_hookean_hessian_x(xn, J, mu, lam, vol, psd=True)
    H_pin     = Q_pin
    H_handle  = Q_h
    H_total   = H_elastic + H_pin + H_handle
    return H_total'''))
    c.append(md(
"""## The two solver loops

They differ only in the **search direction**. Both use a backtracking line
search (tutorial 6). Each records per-iteration metrics and the deformed state."""))
    c.append(code(
'''def run_solver(direction, n_iters=60):
    """direction(g, H) -> step. Returns (states, metrics)."""
    x = X.flatten().reshape(-1, 1).copy()
    states, E, G, DEC = [x.reshape(n, dim).copy()], [], [], []
    for _ in range(n_iters):
        g, H = gradient(x), hessian(x)
        dx = direction(g, H)
        dec = float(np.sqrt(max(-(g.T @ dx)[0, 0], 0.0)))     # sqrt(g^T H^-1 g) for Newton
        alpha, _, _ = simkit.backtracking_line_search(energy, x, g, dx)
        x = x + alpha * dx
        E.append(energy(x)); G.append(float(np.linalg.norm(g))); DEC.append(dec)
        states.append(x.reshape(n, dim).copy())
        if np.linalg.norm(alpha * dx) < 1e-9:
            break
    return states, {"energy": E, "grad": G, "decrement": DEC}

newton_dir = lambda g, H: sp.sparse.linalg.spsolve(H.tocsc(), -g).reshape(-1, 1)
gd_dir     = lambda g, H: -g.reshape(-1, 1)

newton_states, newton_metrics = run_solver(newton_dir)
gd_states,     gd_metrics     = run_solver(gd_dir)
print(f"Newton:           {len(newton_metrics['energy'])} iters, final |grad| = {newton_metrics['grad'][-1]:.2e}")
print(f"Gradient Descent: {len(gd_metrics['energy'])} iters, final |grad| = {gd_metrics['grad'][-1]:.2e}")'''))
    c.append(md(
"""## Convergence curves

Newton's quadratic convergence crushes the gradient norm in a handful of steps;
gradient descent crawls on the stiff, ill-conditioned elastic Hessian. (GD never
forms \\(H\\), so it has no Newton decrement.)"""))
    c.append(code(
'''fig, _ = utils.convergence_plot(
    {"Newton": newton_metrics, "Gradient Descent": gd_metrics},
    title="Newton vs. Gradient Descent on the same problem")
plt.show()'''))
    c.append(md("""## Watching the beam converge (one frame per iteration)"""))
    c.append(code(
'''lims = ((-1.2, 1.8), (-1.3, 0.6))
for name, states, fps in [("newton", newton_states, 6), ("gradient_descent", gd_states, 10)]:
    fig, anim = utils.animate_mesh(states, T, lims=lims, title=name.replace("_", " ").title(),
        pin_pts=X[pin_idx], handle_traj=[s[right_idx] for s in states],
        target_pts=[targets] * len(states), fps=fps)
    utils.save_anim(anim, f"media/05_{name}.mp4", fps=fps)
    plt.close(fig)
display_newton = utils.animate_mesh(newton_states, T, lims=lims, title="Newton's method",
    pin_pts=X[pin_idx], handle_traj=[s[right_idx] for s in newton_states],
    target_pts=[targets] * len(newton_states), fps=6)
plt.close(display_newton[0])
utils.show_anim(display_newton[1])'''))
    c.append(code(
'''gd_anim = utils.animate_mesh(gd_states, T, lims=lims, title="Gradient descent",
    pin_pts=X[pin_idx], handle_traj=[s[right_idx] for s in gd_states],
    target_pts=[targets] * len(gd_states), fps=10)
plt.close(gd_anim[0])
utils.show_anim(gd_anim[1])'''))
    c.append(md(
"""### Takeaways
* **Newton** uses curvature and converges *quadratically* &mdash; ideal for stiff
  elasticity.
* **Gradient descent** ignores curvature and stalls.
* The **Newton decrement** is a clean, scale-aware stopping measure.

But the Newton *step length* needs care &mdash; that's tutorial 6."""))
    save(c, "05_numerical_solver.ipynb")


# =============================================================================
# 06 - Line search
# =============================================================================
def t06():
    c = []
    c.append(md(
"""# 6 &middot; Why Line Search Matters

A Newton *direction* \\(-H^{-1}\\nabla E\\) points the right way, but the *step
length* still matters. Too large a step overshoots, inverts elements, and the
energy **explodes** instead of decreasing.

**Backtracking line search** starts from a full step and shrinks it until the
energy actually decreases (the Armijo condition). To make the failure easy to
watch, we move the handle **slowly right and back** and take a few Newton
iterations per frame &mdash; once with a fixed, too-large step, once with line
search."""))
    c.append(code(BOOT_SIM))
    c.append(md(
"""## A thick, stiff, nearly-incompressible beam

Same objective as tutorial 5 but **thicker** (so bending stores more energy) and
**nearly incompressible** (\\(\\nu = 0.49\\) &mdash; resisting volume change makes
the energy much stiffer), so an over-eager step bites fast and the value of line
search is obvious. The handle (the full right edge) follows a there-and-back
ramp."""))
    c.append(code(
'''X, T = utils.triangulated_grid(nx=16, ny=7, width=2.0, height=0.6)   # thick beam
n, dim = X.shape
J, vol = simkit.deformation_jacobian(X, T), simkit.volume(X, T)
mu, lam = simkit.ympr_to_lame(50.0, 0.49)                            # high Poisson ratio
mu  = np.full((len(T), 1), mu)
lam = np.full((len(T), 1), lam)

pin_idx   = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
right_idx = np.where(X[:, 0] >= X[:, 0].max() - 1e-6)[0]
Q_pin, b_pin = simkit.dirichlet_penalty(pin_idx, X[pin_idx], n, 1e6)

def quad(Q, b, xc):  return 0.5 * (xc.T @ (Q @ xc))[0, 0] + (b.T @ xc)[0, 0]

def make_objective(Q_h, b_h):
    def energy(x):
        xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
        E_elastic = energies.macklin_mueller_neo_hookean_energy_x(xn, J, mu, lam, vol)
        E_pin     = quad(Q_pin, b_pin, xc)
        E_handle  = quad(Q_h, b_h, xc)
        E_total   = E_elastic + E_pin + E_handle
        return E_total
    def gradient(x):
        xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
        g_elastic = energies.macklin_mueller_neo_hookean_gradient_x(xn, J, mu, lam, vol)
        g_pin     = Q_pin @ xc + b_pin
        g_handle  = Q_h @ xc + b_h
        g_total   = g_elastic + g_pin + g_handle
        return g_total
    def hessian(x):
        xn = x.reshape(-1, dim)
        H_elastic = energies.macklin_mueller_neo_hookean_hessian_x(xn, J, mu, lam, vol, psd=True)
        H_pin     = Q_pin
        H_handle  = Q_h
        H_total   = H_elastic + H_pin + H_handle
        return H_total
    return energy, gradient, hessian

ramp = np.concatenate([np.linspace(0, 1, 22), np.linspace(1, 0, 22)])    # right then back
offsets = [np.array([1.0 * r, 0.0]) for r in ramp]'''))
    c.append(md(
"""## The quasi-static driver

At each handle position we take a few Newton iterations. The only difference
between the two runs is the step: a fixed `STEP` (deliberately too big) versus
the backtracking step that guarantees the energy decreases."""))
    c.append(code(
'''DIVERGE = 1e3

def drive(use_line_search, STEP=2.0, iters_per_frame=2):
    x = X.flatten().reshape(-1, 1).copy()
    states, blew_at = [], None
    for k, off in enumerate(offsets):
        Q_h, b_h = simkit.dirichlet_penalty(right_idx, X[right_idx] + off, n, 1e6)
        energy, gradient, hessian = make_objective(Q_h, b_h)
        for _ in range(iters_per_frame):
            g, H = gradient(x), hessian(x)
            dx = sp.sparse.linalg.spsolve(H.tocsc(), -g).reshape(-1, 1)
            alpha = simkit.backtracking_line_search(energy, x, g, dx)[0] if use_line_search else STEP
            x = x + alpha * dx
        states.append(x.reshape(n, dim).copy())
        if not np.isfinite(np.abs(x).max()) or np.abs(x).max() > DIVERGE:
            blew_at = k; break
    return states, blew_at

fixed_states, blew = drive(use_line_search=False, STEP=2.0)
ls_states,    _    = drive(use_line_search=True)
print("fixed step (no line search): blew up at frame", blew, "of", len(offsets))
print("line search: completed all", len(ls_states), "frames")'''))
    c.append(md("""## Without line search: the solver explodes (fixed step = 2x the Newton step)"""))
    c.append(code(
'''lims = ((-1.3, 2.2), (-1.6, 1.6))
fig, anim = utils.animate_mesh(fixed_states, T, lims=lims,
    title="Fixed step, NO line search  ->  explodes", pin_pts=X[pin_idx],
    handle_traj=[s[right_idx] for s in fixed_states], fps=12)
utils.save_anim(anim, "media/06_no_line_search.mp4", fps=12)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md("""## With line search: smooth and stable"""))
    c.append(code(
'''fig, anim = utils.animate_mesh(ls_states, T, lims=lims,
    title="Backtracking line search  ->  stable", pin_pts=X[pin_idx],
    handle_traj=[s[right_idx] for s in ls_states], fps=20)
utils.save_anim(anim, "media/06_line_search.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""### Takeaways
* A good Newton *direction* isn't enough &mdash; an unchecked **step length** can
  diverge.
* **Backtracking line search** shrinks the step until the energy decreases.
* Cheap insurance: a few extra energy evaluations buy a solver that won't blow up."""))
    save(c, "06_line_search.ipynb")


# =============================================================================
# 07 - Time integration
# =============================================================================
def t07():
    c = []
    c.append(md(
"""# 7 &middot; Time Integration: Forward Euler, Backward Euler, BDF2

To animate motion we march the equations of motion in time. The **integrator**
trades accuracy against stability and damping. We compare three on the *same*
cantilever beam:

| scheme | type | order | character |
|---|---|---|---|
| **Forward Euler** | explicit | 1 | cheap, but explodes unless dt is tiny |
| **Backward Euler** | implicit | 1 | rock-solid, but adds numerical damping |
| **BDF2** | implicit | 2 | second-order accurate, barely damps |

Each implicit step minimizes the potential plus an **inertia** term
\\(\\tfrac{c}{2h^2}\\lVert x - \\tilde x\\rVert_M^2\\) pulling \\(x\\) toward where
inertia says it should go (\\(\\tilde x\\)). We write the three steppers explicitly
so the only thing that changes is \\(\\tilde x\\) and \\(c\\)."""))
    c.append(code(BOOT_SIM))
    c.append(md(
"""## The beam and a readable potential

`pot_*` are the static potential (elastic + pinned left edge - gravity). Each
integrator adds its own inertia term on top."""))
    c.append(code(
'''X, T = utils.triangulated_grid(nx=14, ny=4, width=2.0, height=0.3)
n, dim = X.shape
J, vol = simkit.deformation_jacobian(X, T), simkit.volume(X, T)
M_n = simkit.massmatrix(X, T, rho=1.0)
M   = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()
M_lumped = np.asarray(M.sum(1)).flatten()                 # for explicit Forward Euler
f_g = simkit.gravity_force(X, T, a=-9.8, rho=1.0).reshape(-1, 1)
mu, lam = simkit.ympr_to_lame(500.0, 0.4)
mu  = np.full((len(T), 1), mu)
lam = np.full((len(T), 1), lam)
pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
Q_pin, b_pin = simkit.dirichlet_penalty(pin_idx, X[pin_idx], n, 1e7)

def pot_energy(x):
    xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
    E_elastic = energies.macklin_mueller_neo_hookean_energy_x(xn, J, mu, lam, vol)
    E_pin     = 0.5 * (xc.T @ (Q_pin @ xc))[0, 0] + (b_pin.T @ xc)[0, 0]
    E_gravity = -(f_g.T @ xc)[0, 0]
    E_total   = E_elastic + E_pin + E_gravity
    return E_total
def pot_gradient(x):
    xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
    g_elastic = energies.macklin_mueller_neo_hookean_gradient_x(xn, J, mu, lam, vol)
    g_pin     = Q_pin @ xc + b_pin
    g_gravity = -f_g
    g_total   = g_elastic + g_pin + g_gravity
    return g_total
def pot_hessian(x):
    xn = x.reshape(-1, dim)
    H_elastic = energies.macklin_mueller_neo_hookean_hessian_x(xn, J, mu, lam, vol, psd=True)
    H_pin     = Q_pin
    H_total   = H_elastic + H_pin
    return H_total

def newton(energy, gradient, hessian, x0, iters=30):
    return NewtonSolver(energy, gradient, hessian,
                        NewtonSolverParams(max_iter=iters, do_line_search=True)
                        ).solve(x0.reshape(-1, 1)).reshape(n, dim)'''))
    c.append(md(
"""## The three steppers

Each takes the current state and returns the next. The implicit ones (BE, BDF2)
minimize `pot + inertia` &mdash; and differ *only* in the inertia target
\\(\\tilde x\\) and the constant \\(c\\), so we factor that shared minimize into
`implicit_step`. Forward Euler is a plain explicit update. We carry an explicit
velocity `V` so BDF2 is genuinely second order."""))
    c.append(code(
'''def implicit_step(U, x_tilde, c, h):
    """Minimize  pot(x) + inertia(x),  inertia = c/(2 h^2) ||x - x_tilde||^2_M."""
    def energy(x):
        xc = x.reshape(-1, 1)
        E_pot     = pot_energy(x)
        E_inertia = 0.5 * c * ((xc - x_tilde).T @ (M @ (xc - x_tilde)))[0, 0] / h**2
        E_total   = E_pot + E_inertia
        return E_total
    def gradient(x):
        xc = x.reshape(-1, 1)
        g_pot     = pot_gradient(x)
        g_inertia = c * (M @ (xc - x_tilde)) / h**2
        g_total   = g_pot + g_inertia
        return g_total
    def hessian(x):
        H_pot     = pot_hessian(x)
        H_inertia = c * M / h**2
        H_total   = H_pot + H_inertia
        return H_total
    return newton(energy, gradient, hessian, U)

def fe_step(U, V, h):
    """Forward (explicit) Euler:  a = M^-1 (-grad pot);  x += h v;  v += h a."""
    a = (-pot_gradient(U).flatten() / M_lumped).reshape(n, dim)
    U_next = U + h * V
    V_next = V + h * a
    U_next[pin_idx] = X[pin_idx]; V_next[pin_idx] = 0.0      # hard-clamp the pins
    return U_next, V_next

def be_step(U, V, h):
    """Backward Euler:  inertia target x_tilde = U + h V,  c = 1."""
    x_tilde = (U + h * V).reshape(-1, 1)
    U_next  = implicit_step(U, x_tilde, 1.0, h)
    V_next  = (U_next - U) / h
    return U_next, V_next

def bdf2_step(U, V, U_prev, V_prev, h):
    """BDF2:  x_tilde = 4/3 U - 1/3 U_prev + 8h/9 V - 2h/9 V_prev,  c = 9/4."""
    x_tilde = ((4/3) * U - (1/3) * U_prev + (8*h/9) * V - (2*h/9) * V_prev).reshape(-1, 1)
    U_next  = implicit_step(U, x_tilde, 9.0/4.0, h)
    V_next  = (3 * U_next - 4 * U + U_prev) / (2 * h)
    return U_next, V_next'''))
    c.append(md(
"""## Drivers that snapshot at fixed times

So we can compare runs with very different timesteps on the same time axis."""))
    c.append(code(
'''def simulate_fe(h, T_total, n_frames=60):
    sample = np.linspace(0, T_total, n_frames)
    U, V, t, si, frames = X.copy(), np.zeros_like(X), 0.0, 0, [X.copy()]
    while si < n_frames - 1:
        U, V = fe_step(U, V, h); t += h
        if not np.isfinite(U).all() or np.abs(U).max() > 1e3:        # exploded
            frames += [U.copy()] * (n_frames - len(frames)); break
        while si < n_frames - 1 and t >= sample[si + 1]:
            si += 1; frames.append(U.copy())
    return frames

def simulate_be(h, T_total, n_frames=60):
    sample = np.linspace(0, T_total, n_frames)
    U, V, t, si, frames = X.copy(), np.zeros_like(X), 0.0, 0, [X.copy()]
    while si < n_frames - 1:
        U, V = be_step(U, V, h); t += h
        while si < n_frames - 1 and t >= sample[si + 1]:
            si += 1; frames.append(U.copy())
    return frames

def simulate_bdf2(h, T_total, n_frames=60):
    sample = np.linspace(0, T_total, n_frames)
    U, V = X.copy(), np.zeros_like(X)
    U_next, V_next = be_step(U, V, h)                  # bootstrap one BE step
    U_prev, V_prev, U, V = U, V, U_next, V_next
    t, si, frames = h, 0, [X.copy()]
    while si < n_frames - 1:
        U_next, V_next = bdf2_step(U, V, U_prev, V_prev, h)
        U_prev, V_prev, U, V = U, V, U_next, V_next; t += h
        while si < n_frames - 1 and t >= sample[si + 1]:
            si += 1; frames.append(U.copy())
    return frames'''))
    c.append(md(
"""## Order of accuracy, measured on *our* beam

No toy oscillator: we run the beam to a fixed time and compare each scheme to a
**fine-timestep BDF2 reference** on the same mesh. To measure order cleanly we
integrate with **exactly** `round(T/h)` steps so every timestep lands at the
*same* final time (the frame-sampling drivers above would each overshoot `T` by
up to one step). On a log-log plot the error of an order-\\(p\\) method is a line
of slope \\(p\\): **Backward Euler tracks \\(O(h)\\); BDF2 tracks \\(O(h^2)\\)**.
(Forward Euler is excluded &mdash; it needs a far smaller dt just to stay stable,
as we'll see next.)"""))
    c.append(code(
'''def integrate_exact(kind, h, T_total):
    """Step EXACTLY round(T_total / h) times so every dt ends at the same time."""
    steps = int(round(T_total / h))
    U, V = X.copy(), np.zeros_like(X)
    if kind == "Backward Euler":
        for _ in range(steps):
            U, V = be_step(U, V, h)
    else:                                            # BDF2, bootstrapped with one BE step
        U_next, V_next = be_step(U, V, h)
        U_prev, V_prev, U, V = U, V, U_next, V_next
        for _ in range(2, steps + 1):
            U_next, V_next = bdf2_step(U, V, U_prev, V_prev, h)
            U_prev, V_prev, U, V = U, V, U_next, V_next
    return U

T_total = 0.5
reference = integrate_exact("BDF2", 0.0003125, T_total)          # fine BDF2 reference
hs = np.array([0.02, 0.01, 0.005, 0.0025])
errors = {
    "Backward Euler": np.array([np.linalg.norm(integrate_exact("Backward Euler", h, T_total) - reference) for h in hs]),
    "BDF2":           np.array([np.linalg.norm(integrate_exact("BDF2", h, T_total) - reference) for h in hs]),
}
for name, e in errors.items():
    print(f"{name:16s} order ~ {np.polyfit(np.log(hs), np.log(e), 1)[0]:.2f}")

fig, _ = utils.loglog_plot(hs, errors, xlabel="timestep  h", ylabel="error vs fine reference",
    colors=utils.INTEGRATOR_COLORS, ref_slopes={"O(h)": 1, "O(h^2)": 2},
    title="Order of accuracy on the beam")
plt.show()'''))
    c.append(md(
"""## Forward Euler explodes; Backward Euler doesn't (side by side)

Explicit integration is only stable below a tiny critical timestep (CFL). At
\\(h = 0.008\\) &mdash; a timestep Backward Euler handles without blinking
&mdash; **Forward Euler diverges within a few steps** and flies off-screen, while
Backward Euler calmly sags under gravity. This is *why* we use implicit
integrators for stiff elasticity."""))
    c.append(code(
'''h_blow = 0.008
fe_states = simulate_fe(h_blow, 0.3, n_frames=40)
be_states = simulate_be(h_blow, 0.3, n_frames=40)
fig, anim = utils.animate_meshes_grid(
    [{"states": fe_states, "T": T, "title": f"Forward Euler  dt={h_blow}  (EXPLODES)"},
     {"states": be_states, "T": T, "title": f"Backward Euler  dt={h_blow}  (stable)"}],
    lims=((-1.3, 1.3), (-1.1, 0.5)), fps=20,
    suptitle="Same timestep: explicit blows up, implicit is fine")
utils.save_anim(anim, "media/07_fe_vs_be.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## Numerical damping: the timestep changes perceived stiffness

Release the beam under gravity and step **Backward Euler** at three (smooth,
stable) timesteps &mdash; 0.004, 0.008, 0.016. Larger dt &rarr; **more artificial
damping** &rarr; the swing dies out faster, so the material *looks* stiffer even
though the physics is identical."""))
    c.append(code(
'''T_total = 2.0
dts = [0.004, 0.008, 0.016]
be_runs = [{"states": simulate_be(h, T_total, 60), "T": T, "title": f"Backward Euler  dt={h}"}
           for h in dts]
fig, anim = utils.animate_meshes_grid(be_runs, lims=((-1.2, 1.2), (-1.9, 0.5)), fps=30,
    suptitle="Backward Euler: bigger timestep -> more numerical damping")
utils.save_anim(anim, "media/07_be_damping.mp4", fps=30)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## BDF2 barely damps

The same three timesteps with **BDF2**. It keeps swinging at all of them
&mdash; second-order accuracy buys far less numerical dissipation than Backward
Euler."""))
    c.append(code(
'''bdf2_runs = [{"states": simulate_bdf2(h, T_total, 60), "T": T, "title": f"BDF2  dt={h}"}
             for h in dts]
fig, anim = utils.animate_meshes_grid(bdf2_runs, lims=((-1.2, 1.2), (-1.9, 0.5)), fps=30,
    suptitle="BDF2: much less numerical damping at the same timesteps")
utils.save_anim(anim, "media/07_bdf2_damping.mp4", fps=30)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## The damping, quantified: total energy bleeds away faster at bigger dt

The artificial damping isn't just visual &mdash; it removes **mechanical energy**.
We track the total energy (elastic + kinetic + gravitational) over time. For
Backward Euler the bigger the timestep, the **faster the energy drops**; BDF2
(shown for the largest dt) holds onto far more of it."""))
    c.append(code(
'''def total_energy(U, V):
    E_elastic = energies.macklin_mueller_neo_hookean_energy_x(U, J, mu, lam, vol)
    E_kinetic = 0.5 * float(V.flatten() @ (M @ V.flatten()))
    E_gravity = -(f_g.T @ U.reshape(-1, 1))[0, 0]
    E_total   = E_elastic + E_kinetic + E_gravity
    return E_total

def be_energy_trace(h, T_total):
    U, V = X.copy(), np.zeros_like(X)
    ts, Es = [0.0], [total_energy(U, V)]
    for s in range(int(T_total / h)):
        U, V = be_step(U, V, h); ts.append((s + 1) * h); Es.append(total_energy(U, V))
    return np.array(ts), np.array(Es)

def bdf2_energy_trace(h, T_total):
    U, V = X.copy(), np.zeros_like(X)
    Un, Vn = be_step(U, V, h); U_prev, V_prev, U, V = U, V, Un, Vn
    ts, Es = [0.0, h], [total_energy(X, np.zeros_like(X)), total_energy(U, V)]
    for s in range(2, int(T_total / h)):
        Un, Vn = bdf2_step(U, V, U_prev, V_prev, h)
        U_prev, V_prev, U, V = U, V, Un, Vn
        ts.append(s * h); Es.append(total_energy(U, V))
    return np.array(ts), np.array(Es)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
shades = ["#9ecae1", "#4292c6", "#08519c"]
for h, col in zip(dts, shades):
    ts, Es = be_energy_trace(h, 1.5)
    ax.plot(ts, Es, color=col, lw=2, label=f"Backward Euler dt={h}")
ts, Es = bdf2_energy_trace(0.016, 1.5)
ax.plot(ts, Es, color="#2ca02c", lw=2, ls="--", label="BDF2 dt=0.016")
ax.set_xlabel("time (s)"); ax.set_ylabel("total mechanical energy")
ax.set_title("Numerical damping drains energy faster at larger dt")
ax.grid(True, color="0.9"); ax.legend(fontsize=9); plt.show()'''))
    c.append(md(
"""### Takeaways
* **Backward Euler** is order 1; **BDF2** is order 2 (verified on our own beam).
* **Forward Euler** is explicit and explodes unless dt is tiny &mdash; implicit
  methods stay stable at the same dt.
* **Backward Euler** damps motion (and *drains total energy*) more as dt grows;
  **BDF2** stays lively and conserves energy far better."""))
    save(c, "07_time_integration.ipynb")


# =============================================================================
# 08 - Contact ball
# =============================================================================
def t08():
    c = []
    c.append(md(
"""# 8 &middot; Contact: a Ball Pushing into a Soft Block

Contact is just another **energy**: a penalty spring that switches on wherever a
vertex pokes inside an obstacle. We use a ball and the signed distance
\\(\\phi(x) = \\lVert x - c\\rVert - r\\) (negative = inside = penalized).

We do three things:
1. a **static ball probe** &mdash; the block stays frozen while a *ghost* ball
   slides through it, and we read off the contact energy vs the ball's position,
2. a **static ground probe** &mdash; a rigid ball descends into a flat ground and
   we read off the *plane*-contact energy, which peaks when the ball is halfway
   through,
3. a **dynamic simulation** &mdash; a strong ball that visibly dents a hanging,
   pinned block."""))
    c.append(code(BOOT_SIM))
    c.append(md(
"""## The block, the ball, and the contact energy

`contact_springs_sphere_energy(X, k, center, r, M)` sums the penalty over every
vertex inside the ball."""))
    c.append(code(
'''X, T = utils.triangulated_grid(nx=10, ny=12, width=0.6, height=0.8)
X[:, 1] += 0.4                                       # hang it: y in [0, 0.8]
n, dim = X.shape
J, vol = simkit.deformation_jacobian(X, T), simkit.volume(X, T)
M_n = simkit.massmatrix(X, T, rho=1.0)
f_g = simkit.gravity_force(X, T, a=-4.0, rho=1.0).reshape(-1, 1)
mu, lam = simkit.ympr_to_lame(200.0, 0.4)            # soft, so it dents
mu  = np.full((len(T), 1), mu)
lam = np.full((len(T), 1), lam)
pin_idx = np.where(X[:, 1] >= X[:, 1].max() - 1e-6)[0]   # top edge pinned
Q_pin, b_pin = simkit.dirichlet_penalty(pin_idx, X[pin_idx], n, 1e6)

BALL_R = 0.16
K_CONTACT = 5e4                                      # strong contact

def signed_distance(points, center, r):
    return np.linalg.norm(points - np.asarray(center), axis=1) - r'''))
    c.append(md(
"""## Experiment 1 &mdash; static probe: the block does not move

Freeze the block at its rest pose and slide a **ghost ball** up through it. We
evaluate the contact energy of that static block as a function of the ball's
position. The signed-distance background shows the contact region (blue); the
energy on the right rises exactly when the ball overlaps more of the block."""))
    c.append(code(
'''ball_y = np.linspace(-0.5, 0.55, 44)
centers = [np.array([0.0, cy]) for cy in ball_y]
# block is STATIC: evaluate contact energy of the rest block for each ball position
probe_energy = np.array([energies.contact_springs_sphere_energy(X, K_CONTACT, c, BALL_R, M=M_n)
                         for c in centers])
static_block = [X.copy() for _ in centers]           # block never deforms here

fig, anim = utils.animate_dynamics(static_block, T, ball_y, {"contact": probe_energy},
    lims=((-0.6, 0.6), (-0.7, 0.95)), xlabel="ball height", ylabel="contact energy",
    colors=utils.ENERGY_COLORS, ball_centers=centers, ball_radius=BALL_R, sdf=True,
    scene_title="STATIC block, ghost ball sweeping (blue = inside)",
    title="contact energy vs ball position", fps=18)
utils.save_anim(anim, "media/08_ball_probe.mp4", fps=18)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## Experiment 2 &mdash; ground probe: energy peaks halfway through

Same idea, but now the obstacle is the **ground**. We drop a *rigid* ball toward
a flat plane and read off the **plane-contact energy** of the ball's vertices
(`contact_springs_plane_energy`). As the ball sinks in, more of it goes below the
surface and the energy climbs &mdash; reaching its **maximum when the ball is
halfway through** (its center on the ground), then falling as it rises back out.
The blue background is the ground's signed distance \\(\\phi(x) = y - y_{\\text{ground}}\\)
(negative = inside the ground). The ball here is *not* simulated; it is a probe."""))
    c.append(code(
'''ball_X, ball_T = utils.ball_mesh_2d(radius=0.22, n_segments=48)
ball_M = simkit.massmatrix(ball_X, ball_T, rho=1.0)
GROUND_Y = 0.0
floor_p, floor_n = np.array([0.0, GROUND_Y]), np.array([0.0, 1.0])
K_GROUND, R2 = 5e3, 0.22

# sweep the ball center from above the ground down to ground level (halfway
# submerged) and back up; energy peaks at the deepest point
cy = np.concatenate([np.linspace(R2 + 0.2, GROUND_Y, 22), np.linspace(GROUND_Y, R2 + 0.2, 22)])
ball_states = [ball_X + np.array([0.0, y]) for y in cy]
ground_energy = np.array([energies.contact_springs_plane_energy(s, K_GROUND, floor_p, floor_n, M=ball_M)
                          for s in ball_states])
print(f"peak ground-contact energy at ball-center y = {cy[int(np.argmax(ground_energy))]:.2f}  (ground at {GROUND_Y})")

fig, anim = utils.animate_dynamics(ball_states, ball_T, np.arange(len(ball_states)),
    {"contact": ground_energy}, lims=((-0.6, 0.6), (-0.45, 0.7)),
    xlabel="frame", ylabel="ground-contact energy", colors=utils.ENERGY_COLORS,
    floor_y=GROUND_Y, floor_sdf=True, mesh_face=utils.BALL_FACE, mesh_edge=utils.BALL_EDGE,
    scene_title="rigid ball sinking into the ground (blue = inside)",
    title="energy peaks at halfway through", fps=18)
utils.save_anim(anim, "media/08_ground_probe.mp4", fps=18)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""## Experiment 3 &mdash; dynamic simulation: the ball dents the block

Now let the block respond. Each ball position is a Backward-Euler step that
minimizes elastic + pin + gravity + **strong** contact + inertia. The block
visibly deforms as the ball drives up into it."""))
    c.append(code(
'''M = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()
h = 0.02

def be_contact_step(U, U_prev, center):
    x_tilde = (2 * U - U_prev).reshape(-1, 1)         # U + h V, with V=(U-U_prev)/h
    def energy(x):
        xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
        E_elastic = energies.macklin_mueller_neo_hookean_energy_x(xn, J, mu, lam, vol)
        E_pin     = 0.5 * (xc.T @ (Q_pin @ xc))[0, 0] + (b_pin.T @ xc)[0, 0]
        E_gravity = -(f_g.T @ xc)[0, 0]
        E_contact = energies.contact_springs_sphere_energy(xn, K_CONTACT, center, BALL_R, M=M_n)
        E_inertia = 0.5 * ((xc - x_tilde).T @ (M @ (xc - x_tilde)))[0, 0] / h**2
        E_total   = E_elastic + E_pin + E_gravity + E_contact + E_inertia
        return E_total
    def gradient(x):
        xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
        g_elastic = energies.macklin_mueller_neo_hookean_gradient_x(xn, J, mu, lam, vol)
        g_pin     = Q_pin @ xc + b_pin
        g_gravity = -f_g
        g_contact = energies.contact_springs_sphere_gradient(xn, K_CONTACT, center, BALL_R, M=M_n)
        g_inertia = (M @ (xc - x_tilde)) / h**2
        g_total   = g_elastic + g_pin + g_gravity + g_contact + g_inertia
        return g_total
    def hessian(x):
        xn = x.reshape(-1, dim)
        H_elastic = energies.macklin_mueller_neo_hookean_hessian_x(xn, J, mu, lam, vol, psd=True)
        H_pin     = Q_pin
        H_contact = energies.contact_springs_sphere_hessian(xn, K_CONTACT, center, BALL_R, M=M_n)
        H_inertia = M / h**2
        H_total   = H_elastic + H_pin + H_contact + H_inertia
        return H_total
    x = NewtonSolver(energy, gradient, hessian,
                     NewtonSolverParams(max_iter=12, do_line_search=True)).solve(U.flatten().reshape(-1, 1))
    return x.reshape(n, dim)

ball_y = np.concatenate([np.linspace(-0.5, 0.4, 22), np.linspace(0.4, -0.5, 22)])
centers, states, contact_E = [], [], []
U, U_prev = X.copy(), X.copy()
for cy in ball_y:
    c_ball = np.array([0.0, cy])
    U_next = be_contact_step(U, U_prev, c_ball)
    U_prev, U = U, U_next
    centers.append(c_ball); states.append(U.copy())
    contact_E.append(energies.contact_springs_sphere_energy(U, K_CONTACT, c_ball, BALL_R, M=M_n))
print(f"max block displacement: {max(np.abs(s - X).max() for s in states):.3f}")

fig, anim = utils.animate_dynamics(states, T, np.arange(len(states)), {"contact": np.array(contact_E)},
    lims=((-0.6, 0.6), (-0.7, 0.95)), xlabel="frame", ylabel="contact energy",
    colors=utils.ENERGY_COLORS, ball_centers=centers, ball_radius=BALL_R, sdf=True,
    scene_title="ball denting the block (simulated)", title="contact energy", fps=18)
utils.save_anim(anim, "media/08_contact_ball.mp4", fps=18)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""### Takeaways
* **Penalty contact** is an energy that turns on where the signed distance is
  negative; minimizing it pushes penetrating vertices back out.
* The **static probe** shows the energy purely as a function of overlap (no
  simulation); the **dynamic** run lets the block deform in response.
* Deeper penetration &rarr; quadratically larger contact energy."""))
    save(c, "08_contact_ball.ipynb")


# =============================================================================
# 09 - Contact plane (drop)
# =============================================================================
def t09():
    c = []
    c.append(md(
"""# 9 &middot; Contact: Dropping an Elastic Block on the Floor

Inertia + gravity + contact together: an elastic block is **dropped** onto a flat
floor and bounces. We step it with Backward Euler and watch the three energies
trade off &mdash; gravity builds **kinetic** energy on the way down; impact spikes
the **contact** penalty and stores **elastic** energy; the rebound reverses it."""))
    c.append(code(BOOT_SIM))
    c.append(md(
"""## Block, floor, and a Backward-Euler step

`contact_springs_plane_energy(X, k, p, n, M)` penalizes any vertex below the
plane through `p` with upward normal `n`."""))
    c.append(code(
'''X, T = utils.triangulated_grid(nx=10, ny=8, width=0.5, height=0.5)
X[:, 1] += 1.0                                       # start up high
n, dim = X.shape
J, vol = simkit.deformation_jacobian(X, T), simkit.volume(X, T)
M_n = simkit.massmatrix(X, T, rho=1e3)
M   = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()
f_g = simkit.gravity_force(X, T, a=-9.8, rho=1e3).reshape(-1, 1)
mu, lam = simkit.ympr_to_lame(1e5, 0.4)
mu  = np.full((len(T), 1), mu)
lam = np.full((len(T), 1), lam)

FLOOR_Y = -0.6
floor_p, floor_n = np.array([0.0, FLOOR_Y]), np.array([0.0, 1.0])
K_FLOOR, h = 1e5, 0.01

def be_step(U, U_prev):
    x_tilde = (2 * U - U_prev).reshape(-1, 1)
    def energy(x):
        xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
        E_elastic = energies.macklin_mueller_neo_hookean_energy_x(xn, J, mu, lam, vol)
        E_floor   = energies.contact_springs_plane_energy(xn, K_FLOOR, floor_p, floor_n, M=M_n)
        E_gravity = -(f_g.T @ xc)[0, 0]
        E_inertia = 0.5 * ((xc - x_tilde).T @ (M @ (xc - x_tilde)))[0, 0] / h**2
        E_total   = E_elastic + E_floor + E_gravity + E_inertia
        return E_total
    def gradient(x):
        xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
        g_elastic = energies.macklin_mueller_neo_hookean_gradient_x(xn, J, mu, lam, vol)
        g_floor   = energies.contact_springs_plane_gradient(xn, K_FLOOR, floor_p, floor_n, M=M_n)
        g_gravity = -f_g
        g_inertia = (M @ (xc - x_tilde)) / h**2
        g_total   = g_elastic + g_floor + g_gravity + g_inertia
        return g_total
    def hessian(x):
        xn = x.reshape(-1, dim)
        H_elastic = energies.macklin_mueller_neo_hookean_hessian_x(xn, J, mu, lam, vol, psd=True)
        H_floor   = energies.contact_springs_plane_hessian(xn, K_FLOOR, floor_p, floor_n, M=M_n)
        H_inertia = M / h**2
        H_total   = H_elastic + H_floor + H_inertia
        return H_total
    x = NewtonSolver(energy, gradient, hessian,
                     NewtonSolverParams(max_iter=8, do_line_search=True)).solve(U.flatten().reshape(-1, 1))
    return x.reshape(n, dim)'''))
    c.append(md("""## Run the drop, recording elastic / kinetic / contact energy each step"""))
    c.append(code(
'''U, U_prev = X.copy(), X.copy()
states, t_axis = [U.copy()], [0.0]
E_el, E_kin, E_con = [0.0], [0.0], [0.0]
for s in range(130):
    U_next = be_step(U, U_prev)
    v = ((U_next - U) / h).flatten()
    E_el.append(energies.macklin_mueller_neo_hookean_energy_x(U_next, J, mu, lam, vol))
    E_kin.append(0.5 * float(v @ (M @ v)))
    E_con.append(energies.contact_springs_plane_energy(U_next, K_FLOOR, floor_p, floor_n, M=M_n))
    U_prev, U = U, U_next
    states.append(U.copy()); t_axis.append((s + 1) * h)

t_axis = np.array(t_axis)
series = {"elastic": np.array(E_el), "kinetic": np.array(E_kin), "contact": np.array(E_con)}
print(f"impact near t = {t_axis[int(np.argmax(E_con))]:.2f}s")

utils.line_plot(t_axis, series, xlabel="time (s)", ylabel="energy",
                colors=utils.ENERGY_COLORS, title="Energy exchange during the drop")
plt.show()'''))
    c.append(code(
'''fig, anim = utils.animate_dynamics(states, T, t_axis, series, lims=((-1.0, 1.0), (-0.8, 1.4)),
    xlabel="time (s)", ylabel="energy", colors=utils.ENERGY_COLORS, floor_y=FLOOR_Y,
    scene_title="elastic block dropping on the floor",
    title="elastic / kinetic / contact energy", fps=25)
utils.save_anim(anim, "media/09_contact_plane.mp4", fps=25)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md(
"""### Takeaways
* Gravity &rarr; **kinetic** on the way down; impact &rarr; **elastic** +
  **contact**; rebound reverses it.
* The **contact** penalty spikes exactly at impact.
* Penalty contact + Backward Euler lose a little energy each bounce (the spring is
  springy, BE damps), so the block doesn't bounce back to full height."""))
    save(c, "09_contact_plane.ipynb")


# =============================================================================
# 10 - Time complexity
# =============================================================================
def t10():
    c = []
    c.append(md(
"""# 10 &middot; Resolution and Cost: the Accuracy/Speed Trade-off

Refining the mesh improves accuracy but makes every solve more expensive. We
measure both on a cantilever sagging under gravity:

* **cost**: Newton-solve time vs. vertex count (10 resolutions),
* **accuracy**: tip deflection converging to a very fine reference,
* the punchline: a super-fine mesh is accurate but **too slow to iterate on.**"""))
    c.append(code(BOOT_SIM))
    c.append(md(
"""## A timed static solve at any resolution

Soft enough (\\(E = 800\\)) that the sag is large and obvious. `gravity_scale`
lets us ramp gravity for the animation."""))
    c.append(code(
'''import time

def solve_beam(nx, ny, gravity_scale=1.0, iters=30):
    X, T = utils.triangulated_grid(nx, ny, width=2.0, height=0.3)
    n = X.shape[0]
    J, vol = simkit.deformation_jacobian(X, T), simkit.volume(X, T)
    f_g = (gravity_scale * simkit.gravity_force(X, T, a=-9.8, rho=1.0)).reshape(-1, 1)
    mu, lam = simkit.ympr_to_lame(800.0, 0.4)         # soft -> visible sag
    mu  = np.full((len(T), 1), mu)
    lam = np.full((len(T), 1), lam)
    pin = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
    Q_pin, b_pin = simkit.dirichlet_penalty(pin, X[pin], n, 1e7)

    def energy(x):
        xn, xc = x.reshape(-1, 2), x.reshape(-1, 1)
        E_elastic = energies.macklin_mueller_neo_hookean_energy_x(xn, J, mu, lam, vol)
        E_pin     = 0.5 * (xc.T @ (Q_pin @ xc))[0, 0] + (b_pin.T @ xc)[0, 0]
        E_gravity = -(f_g.T @ xc)[0, 0]
        E_total   = E_elastic + E_pin + E_gravity
        return E_total
    def gradient(x):
        xn, xc = x.reshape(-1, 2), x.reshape(-1, 1)
        g_elastic = energies.macklin_mueller_neo_hookean_gradient_x(xn, J, mu, lam, vol)
        g_pin     = Q_pin @ xc + b_pin
        g_gravity = -f_g
        g_total   = g_elastic + g_pin + g_gravity
        return g_total
    def hessian(x):
        xn = x.reshape(-1, 2)
        H_elastic = energies.macklin_mueller_neo_hookean_hessian_x(xn, J, mu, lam, vol, psd=True)
        H_pin     = Q_pin
        H_total   = H_elastic + H_pin
        return H_total

    t0 = time.perf_counter()
    U = NewtonSolver(energy, gradient, hessian,
                     NewtonSolverParams(max_iter=iters, do_line_search=True)).solve(X.flatten().reshape(-1, 1)).reshape(n, 2)
    return {"n": n, "T": T, "X": X, "U": U, "time": time.perf_counter() - t0, "tip_y": U[int(np.argmax(X[:, 0])), 1]}'''))
    c.append(md("""## Cost and accuracy across 10 resolutions"""))
    c.append(code(
'''resolutions = [(5, 2), (7, 3), (10, 4), (14, 5), (19, 6), (26, 9), (34, 11), (44, 15), (56, 19), (70, 23)]
runs = [solve_beam(nx, ny) for nx, ny in resolutions]
reference = solve_beam(90, 30)                        # fine "ground truth"
print(f"fine reference: {reference['n']} verts, {reference['time']*1000:.0f} ms, tip_y = {reference['tip_y']:.4f}")

verts  = np.array([r["n"] for r in runs])
times  = np.array([r["time"] for r in runs])
errors = np.array([abs(r["tip_y"] - reference["tip_y"]) for r in runs])
for r, e in zip(runs, errors):
    print(f"  {r['n']:5d} verts   {r['time']*1000:7.1f} ms   tip_y {r['tip_y']:.4f}   error {e:.2e}")'''))
    c.append(md("""## Compute time vs resolution, and accuracy vs resolution"""))
    c.append(code(
'''fig, axes = plt.subplots(1, 2, figsize=(12, 5))
utils.loglog_plot(verts, {"solve time": times}, xlabel="vertices", ylabel="Newton solve time (s)",
    colors={"solve time": "#d62728"}, ref_slopes={"linear in DOFs": 1},
    title="Compute time grows with resolution", markers=True, ax=axes[0])
utils.loglog_plot(verts, {"tip error": errors}, xlabel="vertices", ylabel="tip deflection error",
    colors={"tip error": "#1f77b4"}, ref_slopes={"O(1/n)": -1},
    title="Accuracy converges to the fine mesh", markers=True, ax=axes[1])
fig.tight_layout(); plt.show()'''))
    c.append(md(
"""## Watching three resolutions sag (clearly!) under gravity

Ramp gravity 0 &rarr; 1 and solve each frame for three resolutions, then **hold at
full gravity** so every beam settles to its final equilibrium and visibly comes to
rest. They converge to nearly the same big droop &mdash; but the finest one took far
longer per frame."""))
    c.append(code(
'''ramp = np.concatenate([np.linspace(0.0, 1.0, 22), np.ones(12)])   # ramp up, then hold at rest
panels = []
for nx, ny, label in [(7, 3, "coarse"), (16, 6, "medium"), (32, 11, "fine")]:
    states = [solve_beam(nx, ny, gravity_scale=g)["U"] for g in ramp]
    panels.append({"states": states, "T": solve_beam(nx, ny)["T"],
                   "title": f"{label}  ({states[0].shape[0]} verts)"})
fig, anim = utils.animate_meshes_grid(panels, lims=((-1.2, 1.2), (-1.8, 0.3)), fps=20,
    suptitle="Same beam, three resolutions, sagging under gravity")
utils.save_anim(anim, "media/10_resolution.mp4", fps=20)
plt.close(fig)
utils.show_anim(anim)'''))
    c.append(md("""## Why "just refine it" is a trap"""))
    c.append(code(
'''good = runs[4]                                            # a mid-resolution mesh
rel = abs(good["tip_y"] - reference["tip_y"]) / abs(reference["tip_y"])
print(f"good-enough ({good['n']:5d} verts): {good['time']*1000:8.1f} ms/solve   ({rel*100:.0f}% tip error)")
print(f"fine        ({reference['n']:5d} verts): {reference['time']*1000:8.1f} ms/solve")
print(f"\\nThe fine mesh is ~{reference['time']/good['time']:.0f}x slower per solve for a tip deflection")
print(f"within {rel*100:.0f}% of it. Interactive editing (many solves/second) is impossible")
print("long before that accuracy is worth it -- pick the coarsest mesh that captures the behavior.")'''))
    c.append(md(
"""### Takeaways
* **Cost** rises steeply with vertex count; each Newton step solves a bigger
  sparse system.
* **Accuracy** improves with resolution but with diminishing returns.
* A super-fine mesh is accurate yet **too slow to iterate on**.

---
That completes the offline SimKit tutorial series (1&ndash;10): building and
understanding the deformation gradient, elastic energies, the elastostatic
minimization, the solvers and line search that drive it, time integration,
contact, and the resolution/cost trade-off."""))
    save(c, "10_time_complexity.ipynb")


if __name__ == "__main__":
    t01(); t02(); t03(); t04(); t05(); t06(); t07(); t08(); t09(); t10()
    print("done")
