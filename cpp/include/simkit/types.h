// Shared type aliases and argument checks for the simkit C++ core.
//
// Conventions (mirroring the NumPy code):
//   * Per-element stacks are row-major 2D matrices with one element per row.
//     A stack of t deformation gradients of size dim x dim is a (t, dim*dim)
//     matrix whose row e is F_e flattened row-major, i.e. F_e(i, j) sits at
//     column i*dim + j. This is exactly what ``F.reshape(t, dim*dim)`` gives
//     for a C-contiguous NumPy array of shape (t, dim, dim).
//   * Hessian blocks w.r.t. the vectorized F use the same row-major
//     vectorization, so a stack of t blocks of size b x b (b = dim*dim) is a
//     (t, b*b) matrix.
//   * Per-element material parameters are vectors of length t, or length 1
//     (broadcast to every element), matching NumPy broadcasting of a (1, 1)
//     array against (t, 1).
#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace simkit {

using Index = std::int64_t;

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXl = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXl = Eigen::Matrix<Index, Eigen::Dynamic, 1>;

using ConstRefRow = Eigen::Ref<const RowMatrixXd>;
using ConstRefRowL = Eigen::Ref<const RowMatrixXl>;
using ConstRefVec = Eigen::Ref<const Eigen::VectorXd>;
using ConstRefVecL = Eigen::Ref<const VectorXl>;

// Fixed-size per-element types. D is the spatial dimension (2 or 3).
template <int D> using MatD = Eigen::Matrix<double, D, D, Eigen::RowMajor>;
template <int D> using VecD = Eigen::Matrix<double, D, 1>;
// Hessian block of a dim x dim quantity w.r.t. the vectorized F: (D*D) x (D*D).
template <int D> using HessD = Eigen::Matrix<double, D * D, D * D, Eigen::RowMajor>;
template <int D> using VecDD = Eigen::Matrix<double, D * D, 1>;

// Per-element parameter lookup with length-1 broadcasting.
inline double param(const ConstRefVec& v, Index e) { return v.size() == 1 ? v[0] : v[e]; }

inline void check_param(const ConstRefVec& v, Index t, const char* name) {
  if (v.size() != 1 && v.size() != t) {
    throw std::invalid_argument(std::string(name) + " must have length 1 or t=" + std::to_string(t) +
                                ", got " + std::to_string(v.size()));
  }
}

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
