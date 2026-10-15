#include "simkit/energies/elastic.h"

#include "simkit/detail/batch.h"
#include "simkit/detail/materials.h"
#include "simkit/psd_project.h"

namespace simkit {

Material material_from_string(const std::string& name) {
  if (name == "linear-elasticity") return Material::linear_elasticity;
  if (name == "arap") return Material::arap;
  if (name == "fcr") return Material::fcr;
  if (name == "macklin-mueller-neo-hookean") return Material::macklin_mueller_neo_hookean;
  if (name == "stvk") return Material::stvk;
  if (name == "neo-hookean") return Material::neo_hookean;
  if (name == "stable-neo-hookean") return Material::stable_neo_hookean;
  throw std::invalid_argument("Unknown material type: " + name);
}

namespace {

// Dispatch helpers. KERNEL(D) names the per-element kernel to call.
#define SIMKIT_DEFINE_MU_LAM_MATERIAL(NAME)                                                                      \
  Eigen::VectorXd NAME##_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam) { \
    const Index t = F.rows();                                                                                     \
    check_param(mu, t, "mu");                                                                                     \
    check_param(lam, t, "lam");                                                                                   \
    if (dim_from_cols(F.cols()) == 2)                                                                             \
      return detail::batch_scalar<2>(F, [&](Index e, const MatD<2>& Fe) { return detail::NAME##_psi<2>(Fe, param(mu, e), param(lam, e)); }); \
    return detail::batch_scalar<3>(F, [&](Index e, const MatD<3>& Fe) { return detail::NAME##_psi<3>(Fe, param(mu, e), param(lam, e)); });   \
  }                                                                                                               \
  RowMatrixXd NAME##_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam) {   \
    const Index t = F.rows();                                                                                     \
    check_param(mu, t, "mu");                                                                                     \
    check_param(lam, t, "lam");                                                                                   \
    if (dim_from_cols(F.cols()) == 2)                                                                             \
      return detail::batch_matrix<2>(F, [&](Index e, const MatD<2>& Fe) { return detail::NAME##_P<2>(Fe, param(mu, e), param(lam, e)); }); \
    return detail::batch_matrix<3>(F, [&](Index e, const MatD<3>& Fe) { return detail::NAME##_P<3>(Fe, param(mu, e), param(lam, e)); });   \
  }                                                                                                               \
  RowMatrixXd NAME##_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam) {    \
    const Index t = F.rows();                                                                                     \
    check_param(mu, t, "mu");                                                                                     \
    check_param(lam, t, "lam");                                                                                   \
    if (dim_from_cols(F.cols()) == 2)                                                                             \
      return detail::batch_hessian<2>(F, [&](Index e, const MatD<2>& Fe) { return detail::NAME##_H<2>(Fe, param(mu, e), param(lam, e)); }); \
    return detail::batch_hessian<3>(F, [&](Index e, const MatD<3>& Fe) { return detail::NAME##_H<3>(Fe, param(mu, e), param(lam, e)); });   \
  }

}  // namespace

// ARAP (mu only).
Eigen::VectorXd arap_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu) {
  check_param(mu, F.rows(), "mu");
  if (dim_from_cols(F.cols()) == 2)
    return detail::batch_scalar<2>(F, [&](Index e, const MatD<2>& Fe) { return detail::arap_psi<2>(Fe, param(mu, e)); });
  return detail::batch_scalar<3>(F, [&](Index e, const MatD<3>& Fe) { return detail::arap_psi<3>(Fe, param(mu, e)); });
}
RowMatrixXd arap_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu) {
  check_param(mu, F.rows(), "mu");
  if (dim_from_cols(F.cols()) == 2)
    return detail::batch_matrix<2>(F, [&](Index e, const MatD<2>& Fe) { return detail::arap_P<2>(Fe, param(mu, e)); });
  return detail::batch_matrix<3>(F, [&](Index e, const MatD<3>& Fe) { return detail::arap_P<3>(Fe, param(mu, e)); });
}
RowMatrixXd arap_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu) {
  check_param(mu, F.rows(), "mu");
  if (dim_from_cols(F.cols()) == 2)
    return detail::batch_hessian<2>(F, [&](Index e, const MatD<2>& Fe) { return detail::arap_H<2>(Fe, param(mu, e)); });
  return detail::batch_hessian<3>(F, [&](Index e, const MatD<3>& Fe) { return detail::arap_H<3>(Fe, param(mu, e)); });
}

SIMKIT_DEFINE_MU_LAM_MATERIAL(fcr)
SIMKIT_DEFINE_MU_LAM_MATERIAL(neo_hookean)
SIMKIT_DEFINE_MU_LAM_MATERIAL(stable_neo_hookean)
SIMKIT_DEFINE_MU_LAM_MATERIAL(stvk)
SIMKIT_DEFINE_MU_LAM_MATERIAL(linear_elasticity)
SIMKIT_DEFINE_MU_LAM_MATERIAL(macklin_mueller_neo_hookean)

#undef SIMKIT_DEFINE_MU_LAM_MATERIAL

Eigen::VectorXd elastic_energy_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam, Material material) {
  switch (material) {
    case Material::linear_elasticity: return linear_elasticity_energy_element_F(F, mu, lam);
    case Material::arap: return arap_energy_element_F(F, mu);
    case Material::fcr: return fcr_energy_element_F(F, mu, lam);
    case Material::macklin_mueller_neo_hookean: return macklin_mueller_neo_hookean_energy_element_F(F, mu, lam);
    case Material::stvk: return stvk_energy_element_F(F, mu, lam);
    case Material::neo_hookean: return neo_hookean_energy_element_F(F, mu, lam);
    case Material::stable_neo_hookean: return stable_neo_hookean_energy_element_F(F, mu, lam);
  }
  throw std::invalid_argument("Unknown material");
}

RowMatrixXd elastic_gradient_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam, Material material) {
  switch (material) {
    case Material::linear_elasticity: return linear_elasticity_gradient_element_F(F, mu, lam);
    case Material::arap: return arap_gradient_element_F(F, mu);
    case Material::fcr: return fcr_gradient_element_F(F, mu, lam);
    case Material::macklin_mueller_neo_hookean: return macklin_mueller_neo_hookean_gradient_element_F(F, mu, lam);
    case Material::stvk: return stvk_gradient_element_F(F, mu, lam);
    case Material::neo_hookean: return neo_hookean_gradient_element_F(F, mu, lam);
    case Material::stable_neo_hookean: return stable_neo_hookean_gradient_element_F(F, mu, lam);
  }
  throw std::invalid_argument("Unknown material");
}

RowMatrixXd elastic_hessian_element_F(const ConstRefRow& F, const ConstRefVec& mu, const ConstRefVec& lam, Material material, bool psd) {
  RowMatrixXd H;
  switch (material) {
    case Material::linear_elasticity: H = linear_elasticity_hessian_element_F(F, mu, lam); break;
    case Material::arap: H = arap_hessian_element_F(F, mu); break;
    case Material::fcr: H = fcr_hessian_element_F(F, mu, lam); break;
    case Material::macklin_mueller_neo_hookean: H = macklin_mueller_neo_hookean_hessian_element_F(F, mu, lam); break;
    case Material::stvk: H = stvk_hessian_element_F(F, mu, lam); break;
    case Material::neo_hookean: H = neo_hookean_hessian_element_F(F, mu, lam); break;
    case Material::stable_neo_hookean: H = stable_neo_hookean_hessian_element_F(F, mu, lam); break;
  }
  if (psd) return psd_project(H, "proj");
  return H;
}

}  // namespace simkit
