// Batched SVD / polar decomposition (simkit.svd_rv, simkit.polar_svd).
#pragma once

#include "simkit/types.h"

namespace simkit {

// Rotation-variant SVD of a stack of dim x dim matrices F (t, dim*dim).
// Outputs U, V (t, dim*dim) and the diagonal factor S (t, dim*dim) as full
// matrices, matching the NumPy function which returns S as a diagonal matrix.
void svd_rv(const ConstRefRow& F, RowMatrixXd& U, RowMatrixXd& S, RowMatrixXd& V);

// Polar decomposition F = R S for a stack of dim x dim matrices.
void polar_svd(const ConstRefRow& F, RowMatrixXd& R, RowMatrixXd& S);

}  // namespace simkit
