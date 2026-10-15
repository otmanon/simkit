// Direct solve of a sparse symmetric system with Eigen's SimplicialLDLT
// (simkit.sparse_solve). Intended for the Newton systems of the integrators,
// which are symmetric and (after PSD projection) positive definite.
#pragma once

#include "simkit/types.h"

namespace simkit {

// A given in CSC form (indptr over columns, row indices, values), n x n;
// only the lower triangle is read. b: (n, k) right-hand sides (row-major).
// Returns false when the factorization fails (the caller should fall back
// to a general solver); x is then left unspecified.
bool sparse_solve_ldlt(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data, Index n,
                       const ConstRefRow& b, RowMatrixXd& x);

}  // namespace simkit
