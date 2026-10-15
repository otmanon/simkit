// Q = J^T blockdiag(He) J, the assembly step shared by every Hessian tier
// (simkit.block_diag_sandwich). He is a (t, b*b) stack of b x b blocks and
// J has t*b rows.
#pragma once

#include "simkit/types.h"

#include <Eigen/Sparse>

namespace simkit {

// Sparse J in CSR form (indptr of length t*b+1, column indices, values) with
// ``ncols`` columns. Produces the COO triplets of the (ncols x ncols) result,
// one block of m_e*m_e entries per element where m_e is the number of
// distinct columns touched by the element's rows. Duplicates are NOT summed.
void block_diag_sandwich_triplets(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data,
                                  Index ncols, const ConstRefRow& He, VectorXl& rows, VectorXl& cols,
                                  Eigen::VectorXd& vals);

// Same, but assembles canonical CSR (sorted column indices, duplicates summed)
// of the (ncols x ncols) result.
void block_diag_sandwich_csr(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data,
                             Index ncols, const ConstRefRow& He, VectorXl& out_indptr, VectorXl& out_indices,
                             Eigen::VectorXd& out_data);

// Dense J (t*b, r): returns the dense (r, r) result.
RowMatrixXd block_diag_sandwich_dense(const ConstRefRow& J, const ConstRefRow& He);

// Convenience for C++ callers holding an Eigen sparse J.
Eigen::SparseMatrix<double> block_diag_sandwich(const Eigen::SparseMatrix<double, Eigen::RowMajor>& J,
                                                const ConstRefRow& He);

}  // namespace simkit
