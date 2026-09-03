"""Energy densities, gradients and Hessians.

Every energy follows the same layered naming convention. The suffix tells you
how much you have to supply yourself, and each layer comes in ``energy`` /
``gradient`` / ``hessian`` flavours. :mod:`simkit.energies.arap` is the
documented reference implementation.

Element tier -- ``*_element_F``, ``*_element_S``, ``*_element_theta``, ``*_element_d``
    Per-element densities and derivative blocks, as stacked arrays. Material
    parameters only: no quadrature weights, no assembly, no global operator.
    This is the only tier that holds the physics.

Global explicit tier -- ``*_x``, ``*_u``, ``*_z``, ``*_S``, ``*_l``
    Takes a prebuilt operator (``J``, ``JB``, ...) plus the quadrature weights
    ``vol``, calls the element tier and assembles. This is what a simulation
    loop calls each step, since the operator and weights are built once and
    reused. The suffix names the primary variable: positions ``x``,
    displacement ``u`` from a reference, reduced coordinates ``z``, symmetric
    stretch ``S``, spring lengths ``l``.

Self-contained tier -- no suffix
    Builds the operator and ``vol`` from rest geometry ``(X, T)`` and forwards
    to the explicit tier. The a-la-carte one-liner for demos and tests.

Which tiers exist
-----------------
``_element_*``, ``_x``, ``_u`` and the self-contained tier are the baseline
that every material provides. The remaining tiers are deliberately partial --
they exist where a paper reproduction needed them, not as a promise:

``_S`` (mixed / stretch)
    ``arap`` and ``macklin_mueller_neo_hookean`` only. Requires an energy
    written in terms of the symmetric stretch rather than ``F``.
``_z`` (reduced coordinates)
    ``elastic`` and ``mass_springs`` only.

:mod:`simkit.energies.elastic` dispatches over materials by name; see its
docstring for the covered set and for the two materials
(``emu``, ``membrane_neo_hookean``) whose signatures put them outside it.
"""

from .elastic import *
from .quadratic import *
from .arap import *
from .fcr import *
from .linear_elasticity import *
from .macklin_mueller_neo_hookean import *
from .stable_neo_hookean import *
from .stvk import *
from .neo_hookean import *
from .kinetic import *
from .contact_springs_plane import *
from .contact_springs_sphere import *
from .emu import *
from .bending_energy import *
from .barrier_energies import *
from .mass_springs import *
