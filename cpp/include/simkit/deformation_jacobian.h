// Deformation Jacobian J mapping flattened vertex positions x (n*dim) to the
// flattened per-element deformation gradients (t*dim*dim), row-major per
// element (simkit.deformation_jacobian).
#pragma once

#include "simkit/types.h"

#include <Eigen/Sparse>

namespace simkit {

// X: rest positions (n, dim); T: simplices (t, dim+1). Fills the COO
// triplets of J, shape (t*dim*dim, n*dim), with nnz = t*dim*dim*(dim+1).
void deformation_jacobian_triplets(const ConstRefRow& X, const ConstRefRowL& T, VectorXl& rows, VectorXl& cols,
                                   Eigen::VectorXd& vals);

// Convenience: assembled sparse J.
Eigen::SparseMatrix<double> deformation_jacobian(const ConstRefRow& X, const ConstRefRowL& T);

}  // namespace simkit
