#include "simkit/polar_svd.h"

#include "simkit/detail/element.h"
#include "simkit/parallel_for.h"

namespace simkit {

namespace {

template <int D>
void svd_rv_impl(const ConstRefRow& F, RowMatrixXd& U, RowMatrixXd& S, RowMatrixXd& V) {
  const Index t = F.rows();
  U.resize(t, D * D);
  S.resize(t, D * D);
  V.resize(t, D * D);
  parallel_for(t, [&](Index e) {
    Eigen::Map<const MatD<D>> Fe(F.row(e).data());
    MatD<D> Ue, Ve;
    VecD<D> s;
    detail::svd_rv<D>(Fe, Ue, s, Ve);
    Eigen::Map<MatD<D>>(U.row(e).data()) = Ue;
    Eigen::Map<MatD<D>>(V.row(e).data()) = Ve;
    Eigen::Map<MatD<D>>(S.row(e).data()) = s.asDiagonal();
  });
}

template <int D>
void polar_svd_impl(const ConstRefRow& F, RowMatrixXd& R, RowMatrixXd& S) {
  const Index t = F.rows();
  R.resize(t, D * D);
  S.resize(t, D * D);
  parallel_for(t, [&](Index e) {
    Eigen::Map<const MatD<D>> Fe(F.row(e).data());
    MatD<D> Re, Se;
    detail::polar<D>(Fe, Re, Se);
    Eigen::Map<MatD<D>>(R.row(e).data()) = Re;
    Eigen::Map<MatD<D>>(S.row(e).data()) = Se;
  });
}

}  // namespace

void svd_rv(const ConstRefRow& F, RowMatrixXd& U, RowMatrixXd& S, RowMatrixXd& V) {
  if (dim_from_cols(F.cols()) == 2)
    svd_rv_impl<2>(F, U, S, V);
  else
    svd_rv_impl<3>(F, U, S, V);
}

void polar_svd(const ConstRefRow& F, RowMatrixXd& R, RowMatrixXd& S) {
  if (dim_from_cols(F.cols()) == 2)
    polar_svd_impl<2>(F, R, S);
  else
    polar_svd_impl<3>(F, R, S);
}

}  // namespace simkit
