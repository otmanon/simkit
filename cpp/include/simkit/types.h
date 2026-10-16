// Shared type aliases and argument checks for the simkit C++ core.
//
// Conventions (mirroring the NumPy code):
//   * Per-element stacks are row-major 2D matrices with one element per row.
//     A stack of t deformation gradients of size dim x dim is a (t, dim*dim)
//     matrix whose row e is F_e flattened row-major, i.e. F_e(i, j) sits at
//     column i*dim + j. This is exactly what ``F.reshape(t, dim*dim)`` gives
//     for a C-contiguous NumPy array of shape (t, dim, dim).
//   * Symmetric blocks (e.g. Hessians w.r.t. the vectorized F) use the same
//     row-major vectorization, so a stack of t blocks of size b x b is a
//     (t, b*b) matrix.
#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace simkit {

using Index = std::int64_t;

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using ConstRefRow = Eigen::Ref<const RowMatrixXd>;

// Fixed-size per-element types. D is the spatial dimension (2 or 3).
template <int D> using MatD = Eigen::Matrix<double, D, D, Eigen::RowMajor>;
template <int D> using VecD = Eigen::Matrix<double, D, 1>;

// Spatial dimension from the number of columns of a flattened F stack.
inline int dim_from_cols(Index cols) {
  if (cols == 4) return 2;
  if (cols == 9) return 3;
  throw std::invalid_argument("expected dim*dim = 4 (2D) or 9 (3D) columns, got " + std::to_string(cols));
}

// Block size b from the number of columns b*b of a flattened Hessian stack.
inline int block_from_cols(Index cols) {
  Index b = 0;
  while (b * b < cols) ++b;
  if (b * b != cols) throw std::invalid_argument("expected b*b columns, got " + std::to_string(cols));
  return static_cast<int>(b);
}

}  // namespace simkit
