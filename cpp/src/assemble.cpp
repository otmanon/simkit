#include "simkit/assemble.h"

#include "simkit/block_diag_sandwich.h"
#include "simkit/energies/elastic.h"
#include "simkit/parallel_for.h"

#include <Eigen/Cholesky>

#include <algorithm>
#include <vector>

namespace simkit {

namespace {

// Split [0, n) into one contiguous range per worker thread.
struct Chunks {
  Index count;
  Index size;
};

Chunks chunks_for(Index n) {
  const Index nt = std::max<Index>(1, std::min<Index>(num_threads(), n));
  return {nt, (n + nt - 1) / nt};
}

}  // namespace

// ---- full space ---------------------------------------------------------------
Eigen::VectorXd assembled_gradient(const SparseCSR& J, const ConstRefRow& P) {
  if (P.size() != J.rows) throw std::invalid_argument("P must have t*b = J.rows entries");
  return csr_rmatvec(J, Eigen::Map<const Eigen::VectorXd>(P.data(), P.size()));
}

void assembled_hessian(const SparseCSR& J, const ConstRefRow& He, VectorXl& out_indptr, VectorXl& out_indices,
                       Eigen::VectorXd& out_data) {
  Eigen::Map<const VectorXl> indptr(J.indptr, J.rows + 1);
  Eigen::Map<const VectorXl> indices(J.indices, J.indptr[J.rows]);
  Eigen::Map<const Eigen::VectorXd> data(J.data, J.indptr[J.rows]);
  block_diag_sandwich_csr(indptr, indices, data, J.cols, He, out_indptr, out_indices, out_data);
}

// ---- dense products -------------------------------------------------------------
Eigen::VectorXd dense_matvec(const ConstRefRow& JB, const ConstRefVec& z) {
  if (z.size() != JB.cols()) throw std::invalid_argument("z must have JB.cols entries");
  const Index n = JB.rows();
  Eigen::VectorXd y(n);
  const Chunks c = chunks_for(n);
  parallel_for(c.count, [&](Index k) {
    const Index r0 = k * c.size, r1 = std::min(n, r0 + c.size);
    if (r0 < r1) y.segment(r0, r1 - r0).noalias() = JB.middleRows(r0, r1 - r0) * z;
  }, 1);
  return y;
}

Eigen::VectorXd dense_rmatvec(const ConstRefRow& JB, const ConstRefVec& p) {
  if (p.size() != JB.rows()) throw std::invalid_argument("p must have JB.rows entries");
  const Index n = JB.rows(), r = JB.cols();
  const Chunks c = chunks_for(n);
  std::vector<Eigen::VectorXd> partial(static_cast<std::size_t>(c.count), Eigen::VectorXd::Zero(r));
  parallel_for(c.count, [&](Index k) {
    const Index r0 = k * c.size, r1 = std::min(n, r0 + c.size);
    if (r0 < r1) partial[k].noalias() = JB.middleRows(r0, r1 - r0).transpose() * p.segment(r0, r1 - r0);
  }, 1);
  Eigen::VectorXd y = Eigen::VectorXd::Zero(r);
  for (const auto& v : partial) y += v;
  return y;
}

// ---- subspace -------------------------------------------------------------------
Eigen::VectorXd assembled_subspace_gradient(const ConstRefRow& JB, const ConstRefRow& P) {
  if (P.size() != JB.rows()) throw std::invalid_argument("P must have t*b = JB.rows entries");
  return dense_rmatvec(JB, Eigen::Map<const Eigen::VectorXd>(P.data(), P.size()));
}

RowMatrixXd assembled_subspace_hessian(const ConstRefRow& JB, const ConstRefRow& He) {
  const int b = block_from_cols(He.cols());
  const Index t = He.rows();
  const Index r = JB.cols();
  if (JB.rows() != t * b) throw std::invalid_argument("JB must have t*b rows");

  // Per thread: a range of elements processed in sub-chunks. When every block
  // of a sub-chunk is positive definite (the usual case after PSD projection)
  // it is factored He = L L^T and the contribution is the symmetric rank
  // update Z^T Z with Z = L^T JB_e, computed on the lower triangle only (half
  // the flops of the general product). Otherwise the sub-chunk uses
  // Q += JB_sub^T (blockdiag(He) JB_sub). The two accumulators are merged at
  // the end.
  const Chunks c = chunks_for(t);
  const Index sub = 512;
  std::vector<RowMatrixXd> partial_full(static_cast<std::size_t>(c.count), RowMatrixXd::Zero(r, r));
  std::vector<Eigen::MatrixXd> partial_lower(static_cast<std::size_t>(c.count), Eigen::MatrixXd::Zero(r, r));
  parallel_for(c.count, [&](Index k) {
    RowMatrixXd& acc_full = partial_full[static_cast<std::size_t>(k)];
    Eigen::MatrixXd& acc_lower = partial_lower[static_cast<std::size_t>(k)];
    RowMatrixXd Y(sub * b, r);
    std::vector<Eigen::MatrixXd> L(static_cast<std::size_t>(sub), Eigen::MatrixXd(b, b));
    const Index e0 = k * c.size, e1 = std::min(t, e0 + c.size);
    for (Index s0 = e0; s0 < e1; s0 += sub) {
      const Index s1 = std::min(e1, s0 + sub);
      const Index rows = (s1 - s0) * b;
      bool all_pd = true;
      for (Index e = s0; e < s1 && all_pd; ++e) {
        Eigen::Map<const RowMatrixXd> Hb(He.row(e).data(), b, b);
        Eigen::LLT<Eigen::MatrixXd> llt(Hb);
        if (llt.info() != Eigen::Success) {
          all_pd = false;
        } else {
          L[static_cast<std::size_t>(e - s0)] = llt.matrixL();
        }
      }
      if (all_pd) {
        for (Index e = s0; e < s1; ++e)
          Y.middleRows((e - s0) * b, b).noalias() =
              L[static_cast<std::size_t>(e - s0)].transpose() * JB.middleRows(e * b, b);
        acc_lower.selfadjointView<Eigen::Lower>().rankUpdate(Y.topRows(rows).transpose());
      } else {
        for (Index e = s0; e < s1; ++e) {
          Eigen::Map<const RowMatrixXd> Hb(He.row(e).data(), b, b);
          Y.middleRows((e - s0) * b, b).noalias() = Hb * JB.middleRows(e * b, b);
        }
        acc_full.noalias() += JB.middleRows(s0 * b, rows).transpose() * Y.topRows(rows);
      }
    }
  }, 1);
  RowMatrixXd Q = RowMatrixXd::Zero(r, r);
  Eigen::MatrixXd lower = Eigen::MatrixXd::Zero(r, r);
  for (Index k = 0; k < c.count; ++k) {
    Q += partial_full[static_cast<std::size_t>(k)];
    lower += partial_lower[static_cast<std::size_t>(k)];
  }
  Q += Eigen::MatrixXd(lower.selfadjointView<Eigen::Lower>());
  return Q;
}

}  // namespace simkit
