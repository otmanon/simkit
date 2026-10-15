"""Build script for simkit.

The pure-Python package needs nothing beyond ``pyproject.toml``. This file
only adds the optional C++ extension ``simkit._cpp`` (pybind11 + Eigen) that
accelerates per-element energies, decompositions and Hessian assembly. The
extension is marked *optional*: when no compiler, pybind11 or Eigen is
available the build prints a warning and the install proceeds with the NumPy
implementations, which are always kept and always importable.

Environment variables:

``SIMKIT_NO_CPP=1``
    Skip the extension entirely.
``EIGEN3_INCLUDE_DIR``
    Directory containing the ``Eigen/`` header tree. If unset, the vendored
    copy under ``cpp/external/eigen`` is used, then common system locations.
"""

from __future__ import annotations

import glob
import os
import sys

from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))


def _find_eigen() -> str | None:
    candidates = []
    env = os.environ.get("EIGEN3_INCLUDE_DIR")
    if env:
        candidates.append(env)
    candidates.append(os.path.join(HERE, "cpp", "external", "eigen"))
    for prefix in filter(None, [os.environ.get("CONDA_PREFIX"), sys.prefix, "/usr/local", "/usr", "/opt/homebrew", "/opt/local"]):
        candidates.append(os.path.join(prefix, "include", "eigen3"))
        candidates.append(os.path.join(prefix, "include"))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "Eigen", "Core")):
            return c
    return None


def _extension_modules():
    if os.environ.get("SIMKIT_NO_CPP"):
        print("simkit: SIMKIT_NO_CPP set, skipping the C++ extension", file=sys.stderr)
        return [], {}
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext
    except ImportError:
        print("simkit: pybind11 not available, skipping the C++ extension", file=sys.stderr)
        return [], {}
    eigen = _find_eigen()
    if eigen is None:
        print(
            "simkit: Eigen headers not found (set EIGEN3_INCLUDE_DIR or restore cpp/external/eigen), "
            "skipping the C++ extension",
            file=sys.stderr,
        )
        return [], {}

    sources = sorted(glob.glob(os.path.join(HERE, "cpp", "src", "**", "*.cpp"), recursive=True))
    sources.append(os.path.join(HERE, "cpp", "bindings", "simkit_cpp.cpp"))
    sources = [os.path.relpath(s, HERE) for s in sources]

    if sys.platform == "win32":
        extra = ["/O2", "/DNDEBUG", "/DEIGEN_MPL2_ONLY"]
    else:
        extra = ["-O3", "-DNDEBUG", "-DEIGEN_MPL2_ONLY"]

    ext = Pybind11Extension(
        "simkit._cpp",
        sources,
        include_dirs=[os.path.relpath(os.path.join(HERE, "cpp", "include"), HERE), eigen],
        cxx_std=17,
        extra_compile_args=extra,
        optional=True,
    )
    return [ext], {"build_ext": build_ext}


_ext_modules, _cmdclass = _extension_modules()

setup(ext_modules=_ext_modules, cmdclass=_cmdclass)
