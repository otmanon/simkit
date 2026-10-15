#include "simkit/sparse_solve.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cstring>
#include <mutex>
#include <vector>

namespace simkit {

namespace {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;
using Solver = Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<Index>>;

// The sparsity pattern of a Newton system is fixed for a given mesh, so the
// ordering and symbolic analysis are reused whenever the CSC structure
// (n, indptr, indices) of the last call repeats. One cached pattern is kept.
std::mutex g_mutex;
Solver g_solver;
std::vector<Index> g_indptr;
std::vector<Index> g_indices;

bool same_pattern(const ConstRefVecL& indptr, const ConstRefVecL& indices) {
  if (g_indptr.size() != static_cast<std::size_t>(indptr.size()) ||
      g_indices.size() != static_cast<std::size_t>(indices.size()))
    return false;
  return std::memcmp(g_indptr.data(), indptr.data(), g_indptr.size() * sizeof(Index)) == 0 &&
         std::memcmp(g_indices.data(), indices.data(), g_indices.size() * sizeof(Index)) == 0;
}

}  // namespace

bool sparse_solve_ldlt(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data, Index n,
                       const ConstRefRow& b, RowMatrixXd& x) {
  if (indptr.size() != n + 1) throw std::invalid_argument("indptr must have n+1 entries");
  if (b.rows() != n) throw std::invalid_argument("b must have n rows");
  if (data.size() != indptr[n]) throw std::invalid_argument("data must have indptr[n] entries");
  // View the CSC arrays directly as an Eigen sparse matrix (no copy).
  Eigen::Map<const SpMat> A(n, n, indptr[n], indptr.data(), indices.data(), data.data());

  std::lock_guard<std::mutex> lock(g_mutex);
  if (!same_pattern(indptr, indices)) {
    g_solver.analyzePattern(A);
    g_indptr.assign(indptr.data(), indptr.data() + indptr.size());
    g_indices.assign(indices.data(), indices.data() + indices.size());
  }
  g_solver.factorize(A);
  if (g_solver.info() != Eigen::Success) {
    g_indptr.clear();  // do not trust the cached analysis after a failure
    g_indices.clear();
    return false;
  }
  Eigen::MatrixXd rhs = b;
  Eigen::MatrixXd sol = g_solver.solve(rhs);
  if (g_solver.info() != Eigen::Success || !sol.allFinite()) return false;
  x = sol;
  return true;
}

}  // namespace simkit
