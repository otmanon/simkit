// Element-tier hyperelastic energies, gradients and Hessians for a stack of
// deformation gradients (simkit.energies.*_element_F).
//
// F is a (t, dim*dim) row-major stack (dim inferred from the column count),
// mu / lam are length t (or length 1, broadcast). Energies return (t),
// gradients (t, dim*dim) and Hessians (t, (dim*dim)^2). No quadrature
// weighting or assembly is applied: that is the job of the global tiers.
#pragma once

#include "simkit/types.h"

#include <string>

namespace simkit {

enum class Material { linear_elasticity, arap, fcr, macklin_mueller_neo_hookean, stvk, neo_hookean, stable_neo_hookean };

// Parse the material names used by simkit.energies.elastic ('arap', 'fcr',
// 'neo-hookean', 'stable-neo-hookean', 'stvk', 'linear-elasticity',
// 'macklin-mueller-neo-hookean'). Throws std::invalid_argument otherwise.
Material material_from_string(const std::string& name);

Eigen::VectorXd elastic_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam, Material material);
RowMatrixXd elastic_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam, Material material);
// psd=true projects every block to the nearest PSD matrix (simkit.psd_project, method 'proj').
RowMatrixXd elastic_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam, Material material, bool psd = true);

// Per-material entry points (same conventions). ARAP takes only mu.
Eigen::VectorXd arap_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu);
RowMatrixXd arap_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu);
RowMatrixXd arap_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu);

Eigen::VectorXd fcr_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd fcr_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd fcr_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);

Eigen::VectorXd neo_hookean_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd neo_hookean_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd neo_hookean_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);

Eigen::VectorXd stable_neo_hookean_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd stable_neo_hookean_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd stable_neo_hookean_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);

Eigen::VectorXd stvk_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd stvk_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd stvk_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);

Eigen::VectorXd linear_elasticity_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd linear_elasticity_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd linear_elasticity_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);

Eigen::VectorXd macklin_mueller_neo_hookean_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd macklin_mueller_neo_hookean_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);
RowMatrixXd macklin_mueller_neo_hookean_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam);

}  // namespace simkit

// ---------------------------------------------------------------------------
// Assembled (global) tiers: energy / gradient / Hessian for an operator J.
//
// The operator maps the flattened primary variable x (positions, displacement
// or reduced coordinates) to flattened per-element deformation gradients:
//   F = J x + Jx0           (Jx0 optional, empty vector = none)
// With a sparse J this is the ``_x`` tier (Jx0 empty) or the ``_u`` tier
// (Jx0 = J x_bar); with a dense J = G J B it is the reduced ``_z`` tier.
// vol holds the per-element quadrature weights (length t or 1).
// ---------------------------------------------------------------------------
#include <Eigen/Sparse>

namespace simkit {

// Read-only view of a CSR matrix (plain aggregate; no ownership).
struct SparseCSR {
  const Index* indptr;
  const Index* indices;
  const double* data;
  Index rows;
  Index cols;
};

Eigen::VectorXd csr_matvec(const SparseCSR& J, const ConstRefVec& x);   // J x
Eigen::VectorXd csr_rmatvec(const SparseCSR& J, const ConstRefVec& p);  // J^T p

// Sparse operator.
double elastic_energy_x(const SparseCSR& J, const ConstRefVec& x, const ConstRefVec& Jx0, const ConstRefVec& mu,
                        const ConstRefVec& lam, const ConstRefVec& vol, Material material, int dim);
Eigen::VectorXd elastic_gradient_x(const SparseCSR& J, const ConstRefVec& x, const ConstRefVec& Jx0,
                                   const ConstRefVec& mu, const ConstRefVec& lam, const ConstRefVec& vol,
                                   Material material, int dim);
// Hessian as canonical CSR (out_indptr, out_indices, out_data) of size cols x cols.
// psd_before_weighting selects the order of the PSD projection and the vol
// weighting: the per-material ``*_hessian_x`` tiers weight first and then
// project (false), simkit.energies.elastic projects first (true).
void elastic_hessian_x(const SparseCSR& J, const ConstRefVec& x, const ConstRefVec& Jx0, const ConstRefVec& mu,
                       const ConstRefVec& lam, const ConstRefVec& vol, Material material, int dim, bool psd,
                       VectorXl& out_indptr, VectorXl& out_indices, Eigen::VectorXd& out_data,
                       bool psd_before_weighting = false);

// Dense (reduced) operator JB of shape (t*dim*dim, r).
double elastic_energy_z(const ConstRefRow& JB, const ConstRefVec& z, const ConstRefVec& Jx0, const ConstRefVec& mu,
                        const ConstRefVec& lam, const ConstRefVec& vol, Material material, int dim);
Eigen::VectorXd elastic_gradient_z(const ConstRefRow& JB, const ConstRefVec& z, const ConstRefVec& Jx0,
                                   const ConstRefVec& mu, const ConstRefVec& lam, const ConstRefVec& vol,
                                   Material material, int dim);
RowMatrixXd elastic_hessian_z(const ConstRefRow& JB, const ConstRefVec& z, const ConstRefVec& Jx0,
                              const ConstRefVec& mu, const ConstRefVec& lam, const ConstRefVec& vol, Material material,
                              int dim, bool psd, bool psd_before_weighting = false);

// Per-element weighting shared by the tiers: multiplies row e of a stack by vol_e.
void scale_rows(RowMatrixXd& A, const ConstRefVec& vol);

}  // namespace simkit
