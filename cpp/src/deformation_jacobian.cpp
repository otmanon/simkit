#include "simkit/deformation_jacobian.h"

#include "simkit/parallel_for.h"

#include <vector>

namespace simkit {

void deformation_jacobian_triplets(const ConstRefRow& X, const ConstRefRowL& T, VectorXl& rows, VectorXl& cols,
                                   Eigen::VectorXd& vals) {
  const int dim = static_cast<int>(X.cols());
  const int dt = static_cast<int>(T.cols());
  if (dim < 1 || dim > 3) throw std::invalid_argument("X must have 1, 2 or 3 columns");
  if (dt != dim + 1) throw std::invalid_argument("T must have dim+1 columns");
  const Index t = T.rows();
  const Index n = X.rows();
  const Index per = static_cast<Index>(dim) * dim * dt;  // nnz per element
  rows.resize(t * per);
  cols.resize(t * per);
  vals.resize(t * per);

  // H = [-1 ... -1; I] (dt x dim).
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dt, dim);
  H.row(0).setConstant(-1.0);
  H.block(1, 0, dim, dim).setIdentity();

  parallel_for(t, [&](Index e) {
    Eigen::MatrixXd XT(dim, dt);  // rest vertices as columns
    for (int k = 0; k < dt; ++k) {
      const Index v = T(e, k);
      if (v < 0 || v >= n) throw std::out_of_range("T references a vertex outside X");
      XT.col(k) = X.row(v).transpose();
    }
    const Eigen::MatrixXd XH = XT * H;                      // dim x dim
    const Eigen::MatrixXd D = (H * XH.inverse()).transpose();  // dim x dt
    // F_ij = sum_k D(j, k) x(T_k, i)
    Index p = e * per;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        for (int k = 0; k < dt; ++k, ++p) {
          rows[p] = e * dim * dim + i * dim + j;
          cols[p] = T(e, k) * dim + i;
          vals[p] = D(j, k);
        }
  });
}

Eigen::SparseMatrix<double> deformation_jacobian(const ConstRefRow& X, const ConstRefRowL& T) {
  VectorXl rows, cols;
  Eigen::VectorXd vals;
  deformation_jacobian_triplets(X, T, rows, cols, vals);
  const int dim = static_cast<int>(X.cols());
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(vals.size()));
  for (Index p = 0; p < vals.size(); ++p) trips.emplace_back(static_cast<int>(rows[p]), static_cast<int>(cols[p]), vals[p]);
  Eigen::SparseMatrix<double> J(T.rows() * dim * dim, X.rows() * dim);
  J.setFromTriplets(trips.begin(), trips.end());
  return J;
}

}  // namespace simkit
