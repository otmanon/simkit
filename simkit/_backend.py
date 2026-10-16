"""Backend selection between the NumPy implementations and the C++ core.

The optional extension :mod:`simkit._cpp` (built by ``setup.py`` from the
sources under ``cpp/``) accelerates the per-element material kernels, the
batched decompositions (``polar_svd``, ``svd_rv``, ``rotation_gradient_F``,
``psd_project``), the deformation Jacobian and the Hessian assembly. Every
accelerated function keeps its NumPy implementation as the fallback, so
``import simkit`` and every public function work identically without the
extension.

Selection rules:

- If the extension imports, it is used by default.
- ``SIMKIT_BACKEND=python`` in the environment forces the NumPy code at
  import time; ``SIMKIT_BACKEND=cpp`` requests the extension and raises if it
  is missing.
- :func:`set_backend` switches at runtime, e.g. to compare the two.

All of this is plain module state and flat functions: no classes.
"""

from __future__ import annotations

import os

try:  # the compiled extension is optional
    from . import _cpp
except ImportError:  # pragma: no cover - depends on the install
    _cpp = None

_BACKENDS = ("cpp", "python")


def cpp_available() -> bool:
    """Whether the compiled ``simkit._cpp`` extension is importable."""
    return _cpp is not None


def _initial_backend() -> str:
    requested = os.environ.get("SIMKIT_BACKEND", "").strip().lower()
    if requested == "cpp" and _cpp is None:
        raise ImportError(
            "SIMKIT_BACKEND=cpp but the simkit._cpp extension is not installed. "
            "Reinstall with a C++ compiler available (pip install -e .) or unset SIMKIT_BACKEND."
        )
    if requested in _BACKENDS:
        return requested
    return "cpp" if _cpp is not None else "python"


_backend = _initial_backend()


def backend() -> str:
    """Name of the active backend: ``'cpp'`` or ``'python'``."""
    return _backend


def set_backend(name: str) -> str:
    """Select the backend for all accelerated functions.

    Parameters
    ----------
    name : str
        ``'cpp'`` or ``'python'``. Requesting ``'cpp'`` without the extension
        installed raises :class:`RuntimeError`.

    Returns
    -------
    name : str
        The backend now active.
    """
    global _backend
    name = str(name).lower()
    if name not in _BACKENDS:
        raise ValueError("backend must be 'cpp' or 'python', got " + repr(name))
    if name == "cpp" and _cpp is None:
        raise RuntimeError("the simkit._cpp extension is not installed; cannot select the 'cpp' backend")
    _backend = name
    return _backend


def use_cpp() -> bool:
    """``True`` when accelerated functions should call into the C++ core.

    This is the single check every accelerated function performs; it is
    cheap (one global read) so it can sit inside hot loops.
    """
    return _backend == "cpp"


def cpp_num_threads(n: int | None = None) -> int:
    """Get or set the worker-thread count of the C++ core.

    Parameters
    ----------
    n : int, optional
        New thread count. ``None`` leaves it unchanged; ``0`` or negative
        restores the default (``SIMKIT_NUM_THREADS`` if set, else the hardware
        concurrency).

    Returns
    -------
    n : int
        The thread count in effect (``1`` when the extension is missing).
    """
    if _cpp is None:
        return 1
    if n is not None:
        _cpp.set_num_threads(int(n))
    return int(_cpp.num_threads())

