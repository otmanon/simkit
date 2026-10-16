// Per-element kernels shared by the batched functions: SVD with rotation
// sign fixing, polar decomposition, rotation gradient, cofactor matrices.
// These are header-only templates on the spatial dimension D (2 or 3) and
// operate on single fixed-size matrices.
#pragma once

#include "simkit/types.h"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>

namespace simkit {
namespace detail {

// Row-major vectorization of a D x D matrix as a (D*D) vector view.
template <int D>
inline Eigen::Map<const VecDD<D>> vec(const MatD<D>& M) {
  return Eigen::Map<const VecDD<D>>(M.data());
}

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

// dR/dF for the rotation of the polar decomposition, as a (D*D) x (D*D)
// block in row-major vectorization (simkit.rotation_gradient_F).
template <int D>
HessD<D> rotation_gradient(const MatD<D>& F);

template <>
inline HessD<2> rotation_gradient<2>(const MatD<2>& F) {
  MatD<2> U, V;
  VecD<2> s;
  svd_rv<2>(F, U, s, V);
  MatD<2> T0;
  T0 << 0, -1, 1, 0;
  const MatD<2> t0m = (1.0 / std::sqrt(2.0)) * U * T0 * V.transpose();
  const VecDD<2> t0 = vec<2>(t0m);
  const double s01 = std::max(s[0] + s[1], 1e-12);
  return (2.0 / s01) * (t0 * t0.transpose());
}

template <>
inline HessD<3> rotation_gradient<3>(const MatD<3>& F) {
  MatD<3> U, V;
  VecD<3> s;
  svd_rv<3>(F, U, s, V);
  MatD<3> T0, T1, T2;
  T0 << 0, -1, 0, 1, 0, 0, 0, 0, 0;
  T1 << 0, 0, 0, 0, 0, 1, 0, -1, 0;
  T2 << 0, 0, 1, 0, 0, 0, -1, 0, 0;
  const double c = 1.0 / std::sqrt(2.0);
  const MatD<3> t0m = c * U * T0 * V.transpose();
  const MatD<3> t1m = c * U * T1 * V.transpose();
  const MatD<3> t2m = c * U * T2 * V.transpose();
  const VecDD<3> t0 = vec<3>(t0m);
  const VecDD<3> t1 = vec<3>(t1m);
  const VecDD<3> t2 = vec<3>(t2m);
  const double s01 = std::max(s[0] + s[1], 1e-8);
  const double s12 = std::max(s[1] + s[2], 1e-8);
  const double s02 = std::max(s[0] + s[2], 1e-8);
  return (2.0 / s01) * (t0 * t0.transpose()) + (2.0 / s12) * (t1 * t1.transpose()) +
         (2.0 / s02) * (t2 * t2.transpose());
}

// Cofactor matrix cof(F) = det(F) F^{-T} = d det(F) / dF.
template <int D>
MatD<D> cofactor(const MatD<D>& F);

template <>
inline MatD<2> cofactor<2>(const MatD<2>& F) {
  MatD<2> C;
  C << F(1, 1), -F(1, 0), -F(0, 1), F(0, 0);
  return C;
}

template <>
inline MatD<3> cofactor<3>(const MatD<3>& F) {
  MatD<3> C;
  C(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
  C(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
  C(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
  C(1, 0) = F(0, 2) * F(2, 1) - F(0, 1) * F(2, 2);
  C(1, 1) = F(0, 0) * F(2, 2) - F(0, 2) * F(2, 0);
  C(1, 2) = F(0, 1) * F(2, 0) - F(0, 0) * F(2, 1);
  C(2, 0) = F(0, 1) * F(1, 2) - F(0, 2) * F(1, 1);
  C(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
  C(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);
  return C;
}

// d vec(cof(F)) / d vec(F) = d^2 det(F) / dF^2 in row-major vectorization.
template <int D>
HessD<D> dcofactor(const MatD<D>& F);

template <>
inline HessD<2> dcofactor<2>(const MatD<2>& /*F*/) {
  // cof = [[F11, -F10], [-F01, F00]]; entry [(i,j),(k,l)] = eps(i,k) eps(j,l).
  HessD<2> H = HessD<2>::Zero();
  H(0 * 2 + 0, 1 * 2 + 1) = 1.0;
  H(0 * 2 + 1, 1 * 2 + 0) = -1.0;
  H(1 * 2 + 0, 0 * 2 + 1) = -1.0;
  H(1 * 2 + 1, 0 * 2 + 0) = 1.0;
  return H;
}

template <>
inline HessD<3> dcofactor<3>(const MatD<3>& F) {
  // entry [(i,j),(k,l)] = sum_{m,n} eps(i,k,m) eps(j,l,n) F(m,n)
  auto eps = [](int a, int b, int c) -> double {
    return static_cast<double>((a - b) * (b - c) * (c - a)) / 2.0;
  };
  HessD<3> H = HessD<3>::Zero();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        if (k == i) continue;
        const int m = 3 - i - k;
        const double eik = eps(i, k, m);
        for (int l = 0; l < 3; ++l) {
          if (l == j) continue;
          const int n = 3 - j - l;
          H(i * 3 + j, k * 3 + l) = eik * eps(j, l, n) * F(m, n);
        }
      }
  return H;
}

}  // namespace detail
}  // namespace simkit
