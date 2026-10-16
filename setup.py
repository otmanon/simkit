"""Build script for simkit.

The pure-Python package needs nothing beyond ``pyproject.toml``. This file
only adds the optional C++ extension ``simkit._cpp`` (pybind11 + Eigen) that
accelerates a few hot spots. The extension is marked *optional*: when no
compiler, pybind11 or Eigen is available the build prints a warning and the
install proceeds with the NumPy implementations, which are always kept and
always importable.

Eigen (header-only) is looked up in this order:

1. ``EIGEN3_INCLUDE_DIR`` (a directory containing ``Eigen/Core``),
2. the git submodule ``cpp/external/eigen`` (``git submodule update --init``),
3. the usual system locations (``/usr/include/eigen3``, ``$CONDA_PREFIX/include/eigen3``, ...),
4. a one-time download of the Eigen 3.4.0 release archive into ``build/eigen``
   (checksum verified). Set ``SIMKIT_NO_DOWNLOAD=1`` to disable this step.

``SIMKIT_NO_CPP=1`` skips the extension entirely.
"""

from __future__ import annotations

import glob
import hashlib
import io
import os
import sys
import tarfile
import urllib.request

from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))

EIGEN_VERSION = "3.4.0"
EIGEN_URL = f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_VERSION}/eigen-{EIGEN_VERSION}.tar.gz"
EIGEN_SHA256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"
EIGEN_CACHE = os.path.join(HERE, "build", "eigen")


def _log(msg: str) -> None:
    print("simkit: " + msg, file=sys.stderr)


def _has_eigen(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "Eigen", "Core"))


def _download_eigen() -> str | None:
    """Fetch the Eigen release archive once into build/eigen and return its include dir."""
    target = os.path.join(EIGEN_CACHE, f"eigen-{EIGEN_VERSION}")
    if _has_eigen(target):
        return target
    if os.environ.get("SIMKIT_NO_DOWNLOAD"):
        return None
    try:
        _log(f"downloading Eigen {EIGEN_VERSION} from {EIGEN_URL}")
        with urllib.request.urlopen(EIGEN_URL, timeout=120) as resp:
            data = resp.read()
    except Exception as exc:  # network, proxy, ...
        _log(f"Eigen download failed ({exc})")
        return None
    digest = hashlib.sha256(data).hexdigest()
    if digest != EIGEN_SHA256:
        _log(f"Eigen archive checksum mismatch ({digest}), ignoring it")
        return None
    try:
        os.makedirs(EIGEN_CACHE, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.startswith(f"eigen-{EIGEN_VERSION}/Eigen/")]
            if hasattr(tarfile, "data_filter"):
                tar.extractall(EIGEN_CACHE, members=members, filter="data")
            else:  # Python < 3.12
                tar.extractall(EIGEN_CACHE, members=members)
    except (OSError, tarfile.TarError) as exc:
        _log(f"could not unpack Eigen into {EIGEN_CACHE} ({exc})")
        return None
    return target if _has_eigen(target) else None


def _find_eigen() -> str | None:
    env = os.environ.get("EIGEN3_INCLUDE_DIR")
    if env and _has_eigen(env):
        return env
    candidates = [os.path.join(HERE, "cpp", "external", "eigen")]
    for prefix in filter(None, [os.environ.get("CONDA_PREFIX"), sys.prefix, "/usr/local", "/usr", "/opt/homebrew", "/opt/local"]):
        candidates.append(os.path.join(prefix, "include", "eigen3"))
        candidates.append(os.path.join(prefix, "include"))
    for c in candidates:
        if _has_eigen(c):
            return c
    return _download_eigen()


# setuptools commands that never compile anything: no point probing for a
# compiler or fetching Eigen when only an sdist or metadata is being produced.
_NO_BUILD_COMMANDS = {"sdist", "egg_info", "dist_info", "clean", "check", "--version", "--help", "-h"}
_BUILD_COMMANDS = {"build", "build_ext", "bdist_wheel", "editable_wheel", "develop", "install", "bdist"}


def _is_metadata_only_invocation() -> bool:
    args = sys.argv[1:]
    if any(a in _BUILD_COMMANDS for a in args):
        return False
    positional = [a for a in args if not a.startswith("-")]
    first = positional[0] if positional else next((a for a in args if a in _NO_BUILD_COMMANDS), None)
    return first in _NO_BUILD_COMMANDS


def _extension_modules():
    if _is_metadata_only_invocation():
        return [], {}
    if os.environ.get("SIMKIT_NO_CPP"):
        _log("SIMKIT_NO_CPP set, skipping the C++ extension")
        return [], {}
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext
    except ImportError:
        _log("pybind11 not available, skipping the C++ extension")
        return [], {}
    eigen = _find_eigen()
    if eigen is None:
        _log("Eigen headers not found (set EIGEN3_INCLUDE_DIR or run 'git submodule update --init'), "
             "skipping the C++ extension")
        return [], {}
    _log(f"building the C++ extension with Eigen from {eigen}")

    sources = sorted(glob.glob(os.path.join(HERE, "cpp", "src", "**", "*.cpp"), recursive=True))
    sources.append(os.path.join(HERE, "cpp", "bindings", "simkit_cpp.cpp"))
    sources = [os.path.relpath(s, HERE) for s in sources]

    # pybind11's helper adds the C++17 switch, /EHsc and /bigobj on MSVC and
    # -fvisibility=hidden elsewhere. The rest: optimization, Eigen's licence
    # guard, and explicit pthread linking on Linux for the std::thread loops.
    link = []
    if sys.platform == "win32":
        extra = ["/O2", "/DNDEBUG", "/DEIGEN_MPL2_ONLY"]
    else:
        extra = ["-O3", "-DNDEBUG", "-DEIGEN_MPL2_ONLY"]
        if sys.platform.startswith("linux"):
            extra.append("-pthread")
            link.append("-pthread")

    ext = Pybind11Extension(
        "simkit._cpp",
        sources,
        include_dirs=[os.path.relpath(os.path.join(HERE, "cpp", "include"), HERE), eigen],
        cxx_std=17,
        extra_compile_args=extra,
        extra_link_args=link,
        optional=True,
    )
    return [ext], {"build_ext": build_ext}


_ext_modules, _cmdclass = _extension_modules()

setup(ext_modules=_ext_modules, cmdclass=_cmdclass)
