// Project a stack of symmetric blocks to positive semi-definite matrices
// (simkit.psd_project).
#pragma once

#include "simkit/types.h"

#include <string>

namespace simkit {

// H: (t, b*b) row-major stack of symmetric b x b blocks (only the lower
// triangle is read, like numpy.linalg.eigh). ``method`` is "proj" (clamp
// eigenvalues below 1e-6 to 1e-6), "abs" (absolute value), or anything else
// (eigenvalues left untouched, i.e. a symmetrization).
RowMatrixXd psd_project(const ConstRefRow& H, const std::string& method = "proj");

}  // namespace simkit
