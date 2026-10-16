# simkit C++ core

A small C++17 library (Eigen only, `std::thread` for parallel loops, no
OpenMP or BLAS) behind the optional Python extension `simkit._cpp`. It keeps
the Python layout: flat free functions in the `simkit` namespace, one header
per Python module.

| Header | Functions |
| --- | --- |
| `simkit/polar_svd.h` | `svd_rv`, `polar_svd` |
| `simkit/psd_project.h` | `psd_project` |
| `simkit/parallel_for.h` | `parallel_for`, `num_threads`, `set_num_threads` |

Per-element stacks are row-major `(t, dim*dim)` matrices, i.e. the flat view
of a C-contiguous NumPy `(t, dim, dim)` array; Hessian blocks use the same
row-major vectorization. The per-element kernels live in
`simkit/detail/element.h` as templates on the spatial dimension.

## Building

From Python, nothing to do: `pip install .` at the repository root compiles
`simkit._cpp` through `setup.py` whenever a C++17 compiler is available
(GCC, Clang or MSVC) and falls back to the pure-Python package otherwise.
Eigen is taken from `EIGEN3_INCLUDE_DIR`, the submodule `external/eigen`,
the system, or a checksummed one-time download into `build/eigen`.

Standalone:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
ctest --test-dir cpp/build
```

This produces the static library `simkit`, the test executable, and, when
`pybind11` is importable from the configured Python, the module
`simkit._cpp` dropped next to the Python package.

## Layout

```
cpp/
  include/simkit/        public headers (one per Python module)
  include/simkit/detail/ per-element kernels
  src/                   batched implementations
  bindings/              pybind11 module simkit._cpp
  tests/                 standalone C++ tests (ctest)
  external/eigen/        Eigen submodule (optional; see setup.py)
```
