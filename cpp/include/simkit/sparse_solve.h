// Direct solve of a sparse symmetric system with Eigen's simplicial Cholesky
// factorizations (simkit.sparse_solve). Intended for the Newton systems of
// the integrators, which are symmetric and (after PSD projection) positive
// definite.
#pragma once

#include "simkit/types.h"

namespace simkit {

enum class SolveMethod { failed = 0, llt = 1, ldlt = 2 };

// A given in CSC form (indptr over columns, row indices, values), n x n;
// only the lower triangle is read. b: (n, k) right-hand sides (row-major).
// Tries SimplicialLLT (Cholesky) first and SimplicialLDLT if that fails;
// returns which one succeeded, or ``failed`` (x then unspecified, the caller
// should fall back to a general solver). The symbolic analysis is cached and
// reused while the sparsity pattern (n, indptr, indices) stays the same.
SolveMethod sparse_solve_cholesky(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data,
                                  Index n, const ConstRefRow& b, RowMatrixXd& x);

// LDLT only (kept for callers that know the matrix is indefinite).
bool sparse_solve_ldlt(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data, Index n,
                       const ConstRefRow& b, RowMatrixXd& x);

}  // namespace simkit
