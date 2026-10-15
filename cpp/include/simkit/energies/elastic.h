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
