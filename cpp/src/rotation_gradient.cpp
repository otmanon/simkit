#include "simkit/rotation_gradient.h"

#include "simkit/detail/element.h"
#include "simkit/parallel_for.h"

namespace simkit {

namespace {
template <int D>
RowMatrixXd impl(const ConstRefRow& F) {
  const Index t = F.rows();
  RowMatrixXd K(t, D * D * D * D);
  parallel_for(t, [&](Index e) {
    Eigen::Map<const MatD<D>> Fe(F.row(e).data());
    Eigen::Map<HessD<D>>(K.row(e).data()) = detail::rotation_gradient<D>(Fe);
  });
  return K;
}
}  // namespace

RowMatrixXd rotation_gradient_F(const ConstRefRow& F) {
  return dim_from_cols(F.cols()) == 2 ? impl<2>(F) : impl<3>(F);
}

}  // namespace simkit
