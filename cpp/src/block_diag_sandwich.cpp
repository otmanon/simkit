#include "simkit/block_diag_sandwich.h"

#include "simkit/assemble.h"

#include "simkit/parallel_for.h"

#include <algorithm>
#include <atomic>
#include <numeric>
#include <utility>
#include <vector>

namespace simkit {

namespace {

// Distinct sorted columns touched by rows [r0, r0+b) of the CSR matrix.
inline void element_columns(const ConstRefVecL& indptr, const ConstRefVecL& indices, Index r0, int b,
                            std::vector<Index>& cols) {
  cols.clear();
  for (Index r = r0; r < r0 + b; ++r)
    for (Index p = indptr[r]; p < indptr[r + 1]; ++p) cols.push_back(indices[p]);
  std::sort(cols.begin(), cols.end());
  cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
}

// Local dense block K_e = Je^T He Je for element e; ``cols`` must already hold
// the element's distinct columns.
inline void element_block(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data,
                          const ConstRefRow& He, Index e, int b, const std::vector<Index>& cols, Eigen::MatrixXd& Je,
                          Eigen::MatrixXd& K) {
  const Index m = static_cast<Index>(cols.size());
  Je.setZero(b, m);
  const Index r0 = e * b;
  for (int r = 0; r < b; ++r)
    for (Index p = indptr[r0 + r]; p < indptr[r0 + r + 1]; ++p) {
      const Index c = std::lower_bound(cols.begin(), cols.end(), indices[p]) - cols.begin();
      Je(r, c) += data[p];
    }
  Eigen::Map<const RowMatrixXd> Hb(He.row(e).data(), b, b);
  K.noalias() = Je.transpose() * Hb * Je;
}

void check_inputs(const ConstRefVecL& indptr, const ConstRefRow& He, int& b, Index& t) {
  b = block_from_cols(He.cols());
  t = He.rows();
  if (indptr.size() != t * b + 1)
    throw std::invalid_argument("J must have t*b rows: expected indptr of length " + std::to_string(t * b + 1) +
                                ", got " + std::to_string(indptr.size()));
}

}  // namespace

void block_diag_sandwich_triplets(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data,
                                  Index /*ncols*/, const ConstRefRow& He, VectorXl& rows, VectorXl& cols,
                                  Eigen::VectorXd& vals) {
  int b;
  Index t;
  check_inputs(indptr, He, b, t);

  // Pass 1: number of distinct columns per element -> triplet offsets.
  std::vector<Index> m(static_cast<std::size_t>(t));
  parallel_for(t, [&](Index e) {
    std::vector<Index> c;
    element_columns(indptr, indices, e * b, b, c);
    m[static_cast<std::size_t>(e)] = static_cast<Index>(c.size());
  });
  std::vector<Index> offset(static_cast<std::size_t>(t) + 1, 0);
  for (Index e = 0; e < t; ++e) offset[e + 1] = offset[e] + m[e] * m[e];
  const Index nnz = offset[t];
  rows.resize(nnz);
  cols.resize(nnz);
  vals.resize(nnz);

  // Pass 2: local blocks scattered to triplets.
  parallel_for(t, [&](Index e) {
    std::vector<Index> c;
    Eigen::MatrixXd Je, K;
    element_columns(indptr, indices, e * b, b, c);
    element_block(indptr, indices, data, He, e, b, c, Je, K);
    const Index me = static_cast<Index>(c.size());
    Index p = offset[e];
    for (Index i = 0; i < me; ++i)
      for (Index j = 0; j < me; ++j, ++p) {
        rows[p] = c[i];
        cols[p] = c[j];
        vals[p] = K(i, j);
      }
  }, 256);
}

void block_diag_sandwich_csr(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data,
                             Index ncols, const ConstRefRow& He, VectorXl& out_indptr, VectorXl& out_indices,
                             Eigen::VectorXd& out_data) {
  int b;
  Index t;
  check_inputs(indptr, He, b, t);

  // Pass 1: distinct columns of every element, kept for the scatter below.
  std::vector<std::vector<Index>> ecols(static_cast<std::size_t>(t));
  parallel_for(t, [&](Index e) { element_columns(indptr, indices, e * b, b, ecols[static_cast<std::size_t>(e)]); });

  // Row buckets: row c receives m_e entries from every element e touching it.
  std::vector<Index> bucket(static_cast<std::size_t>(ncols) + 1, 0);
  for (Index e = 0; e < t; ++e) {
    const auto& c = ecols[static_cast<std::size_t>(e)];
    for (Index r : c) bucket[r + 1] += static_cast<Index>(c.size());
  }
  for (Index r = 0; r < ncols; ++r) bucket[r + 1] += bucket[r];
  const Index raw_nnz = bucket[ncols];
  std::vector<std::atomic<Index>> cursor(static_cast<std::size_t>(ncols));
  for (Index r = 0; r < ncols; ++r) cursor[r].store(bucket[r]);
  VectorXl bcols(raw_nnz);
  Eigen::VectorXd bvals(raw_nnz);

  // Pass 2: local blocks scattered straight into the row buckets.
  parallel_for(t, [&](Index e) {
    const auto& c = ecols[static_cast<std::size_t>(e)];
    Eigen::MatrixXd Je, K;
    element_block(indptr, indices, data, He, e, b, c, Je, K);
    const Index me = static_cast<Index>(c.size());
    for (Index i = 0; i < me; ++i) {
      const Index base = cursor[c[i]].fetch_add(me, std::memory_order_relaxed);
      for (Index j = 0; j < me; ++j) {
        bcols[base + j] = c[j];
        bvals[base + j] = K(i, j);
      }
    }
  }, 256);

  // Pass 3: per row, merge duplicates with a dense accumulator and sort the
  // (few) distinct columns. Row chunks run in parallel with their own scratch.
  std::vector<Index> newcount(static_cast<std::size_t>(ncols), 0);
  const Index nt = std::max<Index>(1, std::min<Index>(num_threads(), ncols));
  const Index chunk = (ncols + nt - 1) / nt;
  parallel_for(nt, [&](Index k) {
    std::vector<double> acc(static_cast<std::size_t>(ncols), 0.0);
    std::vector<Index> mark(static_cast<std::size_t>(ncols), -1);
    std::vector<Index> touched;
    for (Index r = k * chunk; r < std::min(ncols, (k + 1) * chunk); ++r) {
      const Index p0 = bucket[r], p1 = bucket[r + 1];
      touched.clear();
      for (Index p = p0; p < p1; ++p) {
        const Index c = bcols[p];
        if (mark[c] != r) {
          mark[c] = r;
          acc[c] = bvals[p];
          touched.push_back(c);
        } else {
          acc[c] += bvals[p];
        }
      }
      std::sort(touched.begin(), touched.end());
      for (std::size_t i = 0; i < touched.size(); ++i) {
        bcols[p0 + static_cast<Index>(i)] = touched[i];
        bvals[p0 + static_cast<Index>(i)] = acc[touched[i]];
      }
      newcount[r] = static_cast<Index>(touched.size());
    }
  }, 1);

  out_indptr.resize(ncols + 1);
  out_indptr[0] = 0;
  for (Index r = 0; r < ncols; ++r) out_indptr[r + 1] = out_indptr[r] + newcount[r];
  const Index out_nnz = out_indptr[ncols];
  out_indices.resize(out_nnz);
  out_data.resize(out_nnz);
  parallel_for(ncols, [&](Index r) {
    const Index src = bucket[r];
    const Index dst = out_indptr[r];
    for (Index p = 0; p < newcount[r]; ++p) {
      out_indices[dst + p] = bcols[src + p];
      out_data[dst + p] = bvals[src + p];
    }
  }, 4096);
}

RowMatrixXd block_diag_sandwich_dense(const ConstRefRow& J, const ConstRefRow& He) {
  return assembled_subspace_hessian(J, He);
}

Eigen::SparseMatrix<double> block_diag_sandwich(const Eigen::SparseMatrix<double, Eigen::RowMajor>& Jin,
                                                const ConstRefRow& He) {
  Eigen::SparseMatrix<double, Eigen::RowMajor> J = Jin;
  J.makeCompressed();
  const Index nrows = J.rows();
  VectorXl indptr(nrows + 1), indices(J.nonZeros());
  Eigen::VectorXd data(J.nonZeros());
  for (Index r = 0; r <= nrows; ++r) indptr[r] = J.outerIndexPtr()[r];
  for (Index p = 0; p < J.nonZeros(); ++p) {
    indices[p] = J.innerIndexPtr()[p];
    data[p] = J.valuePtr()[p];
  }
  VectorXl rows, cols;
  Eigen::VectorXd vals;
  block_diag_sandwich_triplets(indptr, indices, data, J.cols(), He, rows, cols, vals);
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(vals.size()));
  for (Index p = 0; p < vals.size(); ++p) trips.emplace_back(static_cast<int>(rows[p]), static_cast<int>(cols[p]), vals[p]);
  Eigen::SparseMatrix<double> Q(J.cols(), J.cols());
  Q.setFromTriplets(trips.begin(), trips.end());
  return Q;
}

}  // namespace simkit
