// Per-element material kernels: energy density psi(F), first Piola-Kirchhoff
// stress P = dpsi/dF, and the Hessian d2psi/dF2 as a (D*D) x (D*D) block in
// row-major vectorization. Templated on the spatial dimension D (2 or 3).
//
// The formulas are the ones implemented by the NumPy modules in
// simkit/energies/, written in closed form rather than as generated code.
#pragma once

#include "simkit/detail/element.h"

#include <cmath>

namespace simkit {
namespace detail {

template <int D>
inline HessD<D> identity_hess() {
  return HessD<D>::Identity();
}

// --------------------------------------------------------------------------
// ARAP: psi = mu/2 ||F - R||^2
// --------------------------------------------------------------------------
template <int D>
inline double arap_psi(const MatD<D>& F, double mu) {
  MatD<D> R, S;
  polar<D>(F, R, S);
  return 0.5 * mu * (F - R).squaredNorm();
}

template <int D>
inline MatD<D> arap_P(const MatD<D>& F, double mu) {
  MatD<D> R, S;
  polar<D>(F, R, S);
  return mu * (F - R);
}

template <int D>
inline HessD<D> arap_H(const MatD<D>& F, double mu) {
  return mu * (identity_hess<D>() - rotation_gradient<D>(F));
}

// --------------------------------------------------------------------------
// Fixed corotational: psi = 2 * arap + lam/2 (J - 1)^2
// --------------------------------------------------------------------------
template <int D>
inline double fcr_psi(const MatD<D>& F, double mu, double lam) {
  const double J = F.determinant();
  return 2.0 * arap_psi<D>(F, mu) + 0.5 * lam * (J - 1.0) * (J - 1.0);
}

template <int D>
inline MatD<D> fcr_P(const MatD<D>& F, double mu, double lam) {
  const double J = F.determinant();
  return 2.0 * arap_P<D>(F, mu) + lam * (J - 1.0) * cofactor<D>(F);
}

template <int D>
inline HessD<D> fcr_H(const MatD<D>& F, double mu, double lam) {
  const double J = F.determinant();
  const MatD<D> C = cofactor<D>(F);
  const VecDD<D> c = vec<D>(C);
  return 2.0 * arap_H<D>(F, mu) + lam * (c * c.transpose()) + lam * (J - 1.0) * dcofactor<D>(F);
}

// --------------------------------------------------------------------------
// Neo-Hookean: psi = mu/2 (I_C - D) - mu log J + lam/2 log^2 J
// --------------------------------------------------------------------------
template <int D>
inline double neo_hookean_psi(const MatD<D>& F, double mu, double lam) {
  const double IC = F.squaredNorm();
  const double logJ = std::log(F.determinant());
  return 0.5 * mu * (IC - D) - mu * logJ + 0.5 * lam * logJ * logJ;
}

template <int D>
inline MatD<D> neo_hookean_P(const MatD<D>& F, double mu, double lam) {
  const double logJ = std::log(F.determinant());
  const MatD<D> G = F.inverse().transpose();
  return mu * F + (lam * logJ - mu) * G;
}

template <int D>
inline HessD<D> neo_hookean_H(const MatD<D>& F, double mu, double lam) {
  const double logJ = std::log(F.determinant());
  const MatD<D> G = F.inverse().transpose();
  const VecDD<D> g = vec<D>(G);
  HessD<D> H = mu * identity_hess<D>() + lam * (g * g.transpose());
  const double c = mu - lam * logJ;
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j)
      for (int k = 0; k < D; ++k)
        for (int l = 0; l < D; ++l) H(i * D + j, k * D + l) += c * G(i, l) * G(k, j);
  return H;
}

// --------------------------------------------------------------------------
// Stable Neo-Hookean (Smith et al. 2018):
//   psi = mu/2 (I_C - D) - mu/2 log(I_C + 1) + lam/2 (J - alpha)^2,
//   alpha = 1 + D mu / ((D + 1) lam)
// --------------------------------------------------------------------------
template <int D>
inline double snh_alpha(double mu, double lam) {
  return 1.0 + D * mu / ((D + 1) * lam);
}

template <int D>
inline double stable_neo_hookean_psi(const MatD<D>& F, double mu, double lam) {
  const double IC = F.squaredNorm();
  const double J = F.determinant();
  const double a = snh_alpha<D>(mu, lam);
  return 0.5 * mu * (IC - D) - 0.5 * mu * std::log(IC + 1.0) + 0.5 * lam * (J - a) * (J - a);
}

template <int D>
inline MatD<D> stable_neo_hookean_P(const MatD<D>& F, double mu, double lam) {
  const double IC = F.squaredNorm();
  const double J = F.determinant();
  const double A = mu * IC / (IC + 1.0);
  const double Dd = lam * (J - snh_alpha<D>(mu, lam));
  return A * F + Dd * cofactor<D>(F);
}

template <int D>
inline HessD<D> stable_neo_hookean_H(const MatD<D>& F, double mu, double lam) {
  const double IC = F.squaredNorm();
  const double den = IC + 1.0;
  const double J = F.determinant();
  const double A = mu * IC / den;
  const double B = 2.0 * mu / (den * den);
  const double Dd = lam * (J - snh_alpha<D>(mu, lam));
  const VecDD<D> f = vec<D>(F);
  const MatD<D> C = cofactor<D>(F);
  const VecDD<D> c = vec<D>(C);
  return A * identity_hess<D>() + B * (f * f.transpose()) + lam * (c * c.transpose()) + Dd * dcofactor<D>(F);
}

// --------------------------------------------------------------------------
// St. Venant-Kirchhoff: E = (F^T F - I)/2, psi = mu tr(E^2) + lam/2 tr(E)^2
// --------------------------------------------------------------------------
template <int D>
inline double stvk_psi(const MatD<D>& F, double mu, double lam) {
  const MatD<D> E = 0.5 * (F.transpose() * F - MatD<D>::Identity());
  const double trE = E.trace();
  return mu * E.squaredNorm() + 0.5 * lam * trE * trE;
}

template <int D>
inline MatD<D> stvk_P(const MatD<D>& F, double mu, double lam) {
  const MatD<D> E = 0.5 * (F.transpose() * F - MatD<D>::Identity());
  const MatD<D> S = 2.0 * mu * E + lam * E.trace() * MatD<D>::Identity();
  return F * S;
}

template <int D>
inline HessD<D> stvk_H(const MatD<D>& F, double mu, double lam) {
  const MatD<D> E = 0.5 * (F.transpose() * F - MatD<D>::Identity());
  const MatD<D> FFT = F * F.transpose();
  const double trE = E.trace();
  HessD<D> H;
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j)
      for (int k = 0; k < D; ++k)
        for (int l = 0; l < D; ++l) {
          double v = mu * F(i, l) * F(k, j) + lam * F(i, j) * F(k, l);
          if (i == k) v += 2.0 * mu * E(l, j);
          if (j == l) v += mu * FFT(i, k);
          if (i == k && j == l) v += lam * trE;
          H(i * D + j, k * D + l) = v;
        }
  return H;
}

// --------------------------------------------------------------------------
// Linear elasticity: eps = sym(F) - I, psi = mu ||eps||^2 + lam/2 tr(eps)^2
// --------------------------------------------------------------------------
template <int D>
inline double linear_elasticity_psi(const MatD<D>& F, double mu, double lam) {
  const MatD<D> eps = 0.5 * (F + F.transpose()) - MatD<D>::Identity();
  const double tr = eps.trace();
  return mu * eps.squaredNorm() + 0.5 * lam * tr * tr;
}

template <int D>
inline MatD<D> linear_elasticity_P(const MatD<D>& F, double mu, double lam) {
  const MatD<D> I = MatD<D>::Identity();
  return mu * (F + F.transpose() - 2.0 * I) + lam * (F - I).trace() * I;
}

template <int D>
inline HessD<D> linear_elasticity_H(const MatD<D>& /*F*/, double mu, double lam) {
  HessD<D> H = HessD<D>::Zero();
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j)
      for (int k = 0; k < D; ++k)
        for (int l = 0; l < D; ++l) {
          double v = 0.0;
          if (i == k && j == l) v += mu;
          if (i == l && j == k) v += mu;
          if (i == j && k == l) v += lam;
          H(i * D + j, k * D + l) = v;
        }
  return H;
}

// --------------------------------------------------------------------------
// Macklin-Mueller Neo-Hookean: psi = mu/2 (I_C - D) - mu (J - 1) + lam/2 (J - 1)^2
// --------------------------------------------------------------------------
template <int D>
inline double macklin_mueller_neo_hookean_psi(const MatD<D>& F, double mu, double lam) {
  const double IC = F.squaredNorm();
  const double Jm1 = F.determinant() - 1.0;
  return 0.5 * mu * (IC - D) - mu * Jm1 + 0.5 * lam * Jm1 * Jm1;
}

template <int D>
inline MatD<D> macklin_mueller_neo_hookean_P(const MatD<D>& F, double mu, double lam) {
  const double Jm1 = F.determinant() - 1.0;
  return mu * F + (lam * Jm1 - mu) * cofactor<D>(F);
}

template <int D>
inline HessD<D> macklin_mueller_neo_hookean_H(const MatD<D>& F, double mu, double lam) {
  const double Jm1 = F.determinant() - 1.0;
  const MatD<D> C = cofactor<D>(F);
  const VecDD<D> c = vec<D>(C);
  return mu * identity_hess<D>() + lam * (c * c.transpose()) + (lam * Jm1 - mu) * dcofactor<D>(F);
}

}  // namespace detail
}  // namespace simkit
