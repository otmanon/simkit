// Per-element kernels shared by the batched functions: SVD with rotation
// sign fixing and polar decomposition.
// These are header-only templates on the spatial dimension D (2 or 3) and
// operate on single fixed-size matrices.
#pragma once

#include "simkit/types.h"

#include <Eigen/SVD>


namespace simkit {
namespace detail {

// SVD F = U diag(s) V^T with the "rotation-variant" sign convention of
// simkit.svd_rv: singular values sorted descending, and when exactly one of
// det(U), det(V) is negative the last column of that factor and the last
// singular value flip sign, so that U and V are proper rotations.
template <int D>
inline void svd_rv(const MatD<D>& F, MatD<D>& U, VecD<D>& s, MatD<D>& V) {
  Eigen::JacobiSVD<MatD<D>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  U = svd.matrixU();
  V = svd.matrixV();
  s = svd.singularValues();
  const double det_u = U.determinant();
  const double det_v = V.determinant();
  const double l = det_u * det_v;  // L[D-1, D-1] in the NumPy implementation
  if (det_u < 0 && det_v > 0) U.col(D - 1) *= l;
  if (det_v < 0 && det_u > 0) V.col(D - 1) *= l;
  s[D - 1] *= l;
}

// Polar decomposition F = R S with R in SO(D) and S symmetric.
template <int D>
inline void polar(const MatD<D>& F, MatD<D>& R, MatD<D>& S) {
  MatD<D> U, V;
  VecD<D> s;
  svd_rv<D>(F, U, s, V);
  R.noalias() = U * V.transpose();
  S.noalias() = V * s.asDiagonal() * V.transpose();
}

}  // namespace detail
}  // namespace simkit
