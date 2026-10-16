#include "simkit/psd_project.h"

#include "simkit/parallel_for.h"

#include <Eigen/Eigenvalues>

#include <cmath>

namespace simkit {

namespace {

enum class Method { proj, abs, none };

Method parse_method(const std::string& method) {
  if (method == "proj") return Method::proj;
  if (method == "abs") return Method::abs;
  return Method::none;
}

template <class Mat>
inline void project_block(const Mat& Hin, Method method, Mat& Hout) {
  Eigen::SelfAdjointEigenSolver<Mat> es(Hin);  // reads the lower triangle, like numpy.linalg.eigh
  typename Eigen::SelfAdjointEigenSolver<Mat>::RealVectorType s = es.eigenvalues();
  if (method == Method::proj) {
    for (Index i = 0; i < s.size(); ++i)
      if (s[i] < 1e-6) s[i] = 1e-6;
  } else if (method == Method::abs) {
    s = s.cwiseAbs();
  }
  const auto& U = es.eigenvectors();
  Hout.noalias() = U * s.asDiagonal() * U.transpose();
}

template <int B>
RowMatrixXd fixed_impl(const ConstRefRow& H, Method method) {
  using Mat = Eigen::Matrix<double, B, B, Eigen::RowMajor>;
  const Index t = H.rows();
  RowMatrixXd out(t, B * B);
  parallel_for(t, [&](Index e) {
    Eigen::Map<const Mat> He(H.row(e).data());
    Mat P;
    project_block<Mat>(He, method, P);
    Eigen::Map<Mat>(out.row(e).data()) = P;
  });
  return out;
}

RowMatrixXd dynamic_impl(const ConstRefRow& H, int b, Method method) {
  const Index t = H.rows();
  RowMatrixXd out(t, b * b);
  parallel_for(t, [&](Index e) {
    Eigen::Map<const RowMatrixXd> He(H.row(e).data(), b, b);
    RowMatrixXd P(b, b);
    project_block<RowMatrixXd>(He, method, P);
    Eigen::Map<RowMatrixXd>(out.row(e).data(), b, b) = P;
  });
  return out;
}

}  // namespace

RowMatrixXd psd_project(const ConstRefRow& H, const std::string& method) {
  const int b = block_from_cols(H.cols());
  const Method m = parse_method(method);
  switch (b) {
    case 2: return fixed_impl<2>(H, m);
    case 3: return fixed_impl<3>(H, m);
    case 4: return fixed_impl<4>(H, m);
    case 6: return fixed_impl<6>(H, m);
    case 9: return fixed_impl<9>(H, m);
    default: return dynamic_impl(H, b, m);
  }
}

}  // namespace simkit
