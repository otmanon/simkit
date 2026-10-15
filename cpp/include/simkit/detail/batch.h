// Helpers that lift a per-element kernel to a stack of elements.
#pragma once

#include "simkit/parallel_for.h"
#include "simkit/types.h"

namespace simkit {
namespace detail {

// fn(e, Fe) -> double
template <int D, class Fn>
Eigen::VectorXd batch_scalar(const ConstRefRow& F, Fn fn) {
  const Index t = F.rows();
  Eigen::VectorXd out(t);
  parallel_for(t, [&](Index e) {
    Eigen::Map<const MatD<D>> Fe(F.row(e).data());
    out[e] = fn(e, Fe);
  });
  return out;
}

// fn(e, Fe) -> MatD<D>
template <int D, class Fn>
RowMatrixXd batch_matrix(const ConstRefRow& F, Fn fn) {
  const Index t = F.rows();
  RowMatrixXd out(t, D * D);
  parallel_for(t, [&](Index e) {
    Eigen::Map<const MatD<D>> Fe(F.row(e).data());
    Eigen::Map<MatD<D>>(out.row(e).data()) = fn(e, Fe);
  });
  return out;
}

// fn(e, Fe) -> HessD<D>
template <int D, class Fn>
RowMatrixXd batch_hessian(const ConstRefRow& F, Fn fn) {
  const Index t = F.rows();
  RowMatrixXd out(t, D * D * D * D);
  parallel_for(t, [&](Index e) {
    Eigen::Map<const MatD<D>> Fe(F.row(e).data());
    Eigen::Map<HessD<D>>(out.row(e).data()) = fn(e, Fe);
  });
  return out;
}

}  // namespace detail
}  // namespace simkit
