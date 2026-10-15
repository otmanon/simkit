# Vendored Eigen

This directory holds the `Eigen/` header tree of
[Eigen 3.4.0](https://gitlab.com/libeigen/eigen/-/releases/3.4.0), copied
verbatim from the official release archive
(`https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz`).
Only the `Eigen/` directory is included; `unsupported/`, docs and tests are
left out.

Eigen is licensed under the MPL2 (see `COPYING.MPL2`); a few files carry the
BSD or MINPACK licenses noted in `COPYING.README`.

The simkit build looks for Eigen in this order:

1. the `EIGEN3_INCLUDE_DIR` environment variable,
2. this vendored copy,
3. the usual system locations (`/usr/include/eigen3`, `$CONDA_PREFIX/include/eigen3`, ...).

To use a different Eigen, set `EIGEN3_INCLUDE_DIR` or replace this directory
(for instance with a git submodule of the Eigen repository).
