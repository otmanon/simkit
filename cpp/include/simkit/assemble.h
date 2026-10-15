// Assembly of per-element stresses and Hessian blocks into the global (or
// reduced) gradient and Hessian. Material-agnostic: the element tier
// produces P (t, b) and He (t, b*b); these functions apply the operator.
//
//   full space, sparse J (t*b, m):     g = J^T vec(P),   Q = J^T blockdiag(He) J
//   subspace, dense JB (t*b, r):       g = JB^T vec(P),  Q = JB^T blockdiag(He) JB
//
// (simkit.assembled_gradient, assembled_hessian, assembled_subspace_gradient,
// assembled_subspace_hessian). P and He are the vol-weighted (and, for He,
// PSD-projected) per-element quantities in row-major vectorization.
#pragma once

#include "simkit/types.h"

namespace simkit {

struct SparseCSR;  // simkit/energies/elastic.h

// ---- full space (sparse operator) -------------------------------------------
// g = J^T vec(P); P is (t, b) with t*b == J.rows.
Eigen::VectorXd assembled_gradient(const SparseCSR& J, const ConstRefRow& P);
// Q = J^T blockdiag(He) J as canonical CSR of size cols x cols.
void assembled_hessian(const SparseCSR& J, const ConstRefRow& He, VectorXl& out_indptr, VectorXl& out_indices,
                       Eigen::VectorXd& out_data);

// ---- subspace (dense reduced operator) --------------------------------------
// Both are chunked over element rows and run on the worker threads; the
// Hessian is formed as JB^T (blockdiag(He) JB) with one GEMM per chunk.
Eigen::VectorXd assembled_subspace_gradient(const ConstRefRow& JB, const ConstRefRow& P);
RowMatrixXd assembled_subspace_hessian(const ConstRefRow& JB, const ConstRefRow& He);

// Threaded dense products used by the reduced tiers: y = JB z (rows chunked)
// and y = JB^T p (rows chunked, partial sums reduced).
Eigen::VectorXd dense_matvec(const ConstRefRow& JB, const ConstRefVec& z);
Eigen::VectorXd dense_rmatvec(const ConstRefRow& JB, const ConstRefVec& p);

}  // namespace simkit
