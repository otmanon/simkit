// dR/dF of the polar rotation for a stack of deformation gradients
// (simkit.rotation_gradient_F).
#pragma once

#include "simkit/types.h"

namespace simkit {

// F: (t, dim*dim). Returns (t, (dim*dim)^2) row-major blocks.
RowMatrixXd rotation_gradient_F(const ConstRefRow& F);

}  // namespace simkit
