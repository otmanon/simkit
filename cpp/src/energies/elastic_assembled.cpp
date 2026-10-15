#include "simkit/block_diag_sandwich.h"
#include "simkit/energies/elastic.h"
#include "simkit/parallel_for.h"
#include "simkit/psd_project.h"

#include <stdexcept>

namespace simkit {

Eigen::VectorXd csr_matvec(const SparseCSR& J, const ConstRefVec& x) {
  if (x.size() != J.cols) throw std::invalid_argument("csr_matvec: x must have J.cols entries");
  Eigen::VectorXd y(J.rows);
  parallel_for(J.rows, [&](Index r) {
    double s = 0.0;
    for (Index p = J.indptr[r]; p < J.indptr[r + 1]; ++p) s += J.data[p] * x[J.indices[p]];
    y[r] = s;
  }, 4096);
  return y;
}

Eigen::VectorXd csr_rmatvec(const SparseCSR& J, const ConstRefVec& p) {
  if (p.size() != J.rows) throw std::invalid_argument("csr_rmatvec: p must have J.rows entries");
  Eigen::VectorXd y = Eigen::VectorXd::Zero(J.cols);
  for (Index r = 0; r < J.rows; ++r) {
    const double pr = p[r];
    for (Index q = J.indptr[r]; q < J.indptr[r + 1]; ++q) y[J.indices[q]] += J.data[q] * pr;
  }
  return y;
}

void scale_rows(RowMatrixXd& A, const ConstRefVec& vol) {
  check_param(vol, A.rows(), "vol");
  if (vol.size() == 1) {
    A *= vol[0];
    return;
  }
  parallel_for(A.rows(), [&](Index e) { A.row(e) *= vol[e]; }, 4096);
}

namespace {

// Flattened deformation gradients as a (t, dim*dim) stack.
RowMatrixXd deformation_stack(Eigen::VectorXd&& f, const ConstRefVec& Jx0, int dim) {
  const Index b = static_cast<Index>(dim) * dim;
  if (f.size() % b != 0) throw std::invalid_argument("J must have t*dim*dim rows");
  if (Jx0.size() != 0) {
    if (Jx0.size() != f.size()) throw std::invalid_argument("Jx0 must have the same length as J x");
    f += Jx0;
  }
  const Index t = f.size() / b;
  return Eigen::Map<const RowMatrixXd>(f.data(), t, b);
}

double weighted_energy(const RowMatrixXd& F, const ConstRefVec& mu, const ConstRefVec& lam, const ConstRefVec& vol,
                       Material material) {
  const Eigen::VectorXd psi = elastic_energy_element_F(F, mu, lam, material);
  check_param(vol, psi.size(), "vol");
  if (vol.size() == 1) return vol[0] * psi.sum();
  return vol.dot(psi);
}

RowMatrixXd weighted_stress(const RowMatrixXd& F, const ConstRefVec& mu, const ConstRefVec& lam, const ConstRefVec& vol,
                            Material material) {
  RowMatrixXd P = elastic_gradient_element_F(F, mu, lam, material);
  scale_rows(P, vol);
  return P;
}

RowMatrixXd weighted_hessian(const RowMatrixXd& F, const ConstRefVec& mu, const ConstRefVec& lam,
                             const ConstRefVec& vol, Material material, bool psd, bool psd_before_weighting) {
  RowMatrixXd He = elastic_hessian_element_F(F, mu, lam, material, psd && psd_before_weighting);
  scale_rows(He, vol);
  if (psd && !psd_before_weighting) He = psd_project(He, "proj");
  return He;
}

}  // namespace

// ---- sparse operator ---------------------------------------------------------
double elastic_energy_x(const SparseCSR& J, const ConstRefVec& x, const ConstRefVec& Jx0, const ConstRefVec& mu,
                        const ConstRefVec& lam, const ConstRefVec& vol, Material material, int dim) {
  const RowMatrixXd F = deformation_stack(csr_matvec(J, x), Jx0, dim);
  return weighted_energy(F, mu, lam, vol, material);
}

Eigen::VectorXd elastic_gradient_x(const SparseCSR& J, const ConstRefVec& x, const ConstRefVec& Jx0,
                                   const ConstRefVec& mu, const ConstRefVec& lam, const ConstRefVec& vol,
                                   Material material, int dim) {
  const RowMatrixXd F = deformation_stack(csr_matvec(J, x), Jx0, dim);
  const RowMatrixXd P = weighted_stress(F, mu, lam, vol, material);
  return csr_rmatvec(J, Eigen::Map<const Eigen::VectorXd>(P.data(), P.size()));
}

void elastic_hessian_x(const SparseCSR& J, const ConstRefVec& x, const ConstRefVec& Jx0, const ConstRefVec& mu,
                       const ConstRefVec& lam, const ConstRefVec& vol, Material material, int dim, bool psd,
                       VectorXl& out_indptr, VectorXl& out_indices, Eigen::VectorXd& out_data,
                       bool psd_before_weighting) {
  const RowMatrixXd F = deformation_stack(csr_matvec(J, x), Jx0, dim);
  const RowMatrixXd He = weighted_hessian(F, mu, lam, vol, material, psd, psd_before_weighting);
  Eigen::Map<const VectorXl> indptr(J.indptr, J.rows + 1);
  Eigen::Map<const VectorXl> indices(J.indices, J.indptr[J.rows]);
  Eigen::Map<const Eigen::VectorXd> data(J.data, J.indptr[J.rows]);
  block_diag_sandwich_csr(indptr, indices, data, J.cols, He, out_indptr, out_indices, out_data);
}

// ---- dense (reduced) operator ------------------------------------------------
double elastic_energy_z(const ConstRefRow& JB, const ConstRefVec& z, const ConstRefVec& Jx0, const ConstRefVec& mu,
                        const ConstRefVec& lam, const ConstRefVec& vol, Material material, int dim) {
  if (z.size() != JB.cols()) throw std::invalid_argument("z must have JB.cols entries");
  const RowMatrixXd F = deformation_stack(JB * z, Jx0, dim);
  return weighted_energy(F, mu, lam, vol, material);
}

Eigen::VectorXd elastic_gradient_z(const ConstRefRow& JB, const ConstRefVec& z, const ConstRefVec& Jx0,
                                   const ConstRefVec& mu, const ConstRefVec& lam, const ConstRefVec& vol,
                                   Material material, int dim) {
  if (z.size() != JB.cols()) throw std::invalid_argument("z must have JB.cols entries");
  const RowMatrixXd F = deformation_stack(JB * z, Jx0, dim);
  const RowMatrixXd P = weighted_stress(F, mu, lam, vol, material);
  return JB.transpose() * Eigen::Map<const Eigen::VectorXd>(P.data(), P.size());
}

RowMatrixXd elastic_hessian_z(const ConstRefRow& JB, const ConstRefVec& z, const ConstRefVec& Jx0,
                              const ConstRefVec& mu, const ConstRefVec& lam, const ConstRefVec& vol, Material material,
                              int dim, bool psd, bool psd_before_weighting) {
  if (z.size() != JB.cols()) throw std::invalid_argument("z must have JB.cols entries");
  const RowMatrixXd F = deformation_stack(JB * z, Jx0, dim);
  const RowMatrixXd He = weighted_hessian(F, mu, lam, vol, material, psd, psd_before_weighting);
  return block_diag_sandwich_dense(JB, He);
}

}  // namespace simkit
