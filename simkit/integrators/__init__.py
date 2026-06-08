"""Time integrators.

Each integrator advances a second-order system ``M x'' = -grad V(x)`` by one
timestep from a position history, given the potential's energy / gradient /
(and, for implicit schemes, hessian) callables, the mass matrix, and the
timestep.

* :func:`backward_euler` -- implicit, 1st order, unconditionally stable, damps.
* :func:`bdf2` -- implicit, 2nd order, stable, barely damps.
* :func:`forward_euler` -- explicit, 1st order, cheap, conditionally stable.

The implicit integrators call :func:`~simkit.solvers.newton_solver`
automatically from the supplied potential callables plus an inertial term, and
forward the Newton parameters (``tolerance``, ``max_iter``, ``do_line_search``)
as keyword arguments. The explicit integrator needs neither a hessian nor any
solver information.
"""

from .backward_euler import backward_euler
from .bdf2 import bdf2
from .forward_euler import forward_euler
