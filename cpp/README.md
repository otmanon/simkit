# simkit C++ core

A small C++17 library (Eigen only) that reimplements the hot spots of the
NumPy code: per-element material kernels, batched decompositions, the
deformation Jacobian and the Hessian assembly. It keeps the Python layout,
flat free functions in the `simkit` namespace, one header per Python module:

| Header | Functions |
| --- | --- |
| `simkit/polar_svd.h` | `svd_rv`, `polar_svd` |
| `simkit/rotation_gradient.h` | `rotation_gradient_F` |
| `simkit/psd_project.h` | `psd_project` |
| `simkit/energies/elastic.h` | `<material>_{energy,gradient,hessian}_element_F` for `arap`, `fcr`, `neo_hookean`, `stable_neo_hookean`, `stvk`, `linear_elasticity`, `macklin_mueller_neo_hookean`, plus the `elastic_*_element_F` dispatcher and `material_from_string` |
| `simkit/deformation_jacobian.h` | `deformation_jacobian`, `deformation_jacobian_triplets` |
| `simkit/block_diag_sandwich.h` | `block_diag_sandwich` (sparse / dense / CSR variants) |
| `simkit/parallel_for.h` | `parallel_for`, `num_threads`, `set_num_threads` |

Per-element stacks are row-major `(t, dim*dim)` matrices, i.e. the flat view
of a C-contiguous NumPy `(t, dim, dim)` array, and Hessian blocks use the same
row-major vectorization. The per-element kernels themselves live in
`simkit/detail/materials.h` and `simkit/detail/element.h` as templates on the
spatial dimension, so they can be reused from other C++ code without the
batching layer.

Loops over elements are parallelized with `std::thread` (no OpenMP); the
thread count comes from `SIMKIT_NUM_THREADS` or the hardware concurrency.

## Building

From Python, nothing to do: `pip install .` at the repository root compiles
`simkit._cpp` through `setup.py` whenever a C++17 compiler is available, and
falls back to the pure-Python package otherwise (see the top-level README).

Standalone:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
ctest --test-dir cpp/build        # finite-difference and equivalence checks
```

This produces the static library `simkit`, the test executable, and, when
`pybind11` is importable from the configured Python, the module
`simkit._cpp` dropped next to the Python package. Eigen is resolved from
`EIGEN3_INCLUDE_DIR`, then the vendored copy in `external/eigen`, then the
system (`find_package(Eigen3)`).

## Layout

```
cpp/
  include/simkit/        public headers (one per Python module)
  include/simkit/detail/ per-element kernels and batching helpers
  src/                   batched implementations
  bindings/              pybind11 module simkit._cpp
  tests/                 standalone C++ tests (ctest)
  external/eigen/        vendored Eigen 3.4.0 headers
```
