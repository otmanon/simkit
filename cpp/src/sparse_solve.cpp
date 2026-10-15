#include "simkit/sparse_solve.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cstring>
#include <mutex>
#include <vector>

namespace simkit {

namespace {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;
using LLT = Eigen::SimplicialLLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<Index>>;
using LDLT = Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<Index>>;

// The sparsity pattern of a Newton system is fixed for a given mesh, so the
// ordering and symbolic analysis are reused whenever the CSC structure
// (n, indptr, indices) of the last call repeats. One cached pattern is kept,
// with the analysis of each factorization done lazily on first use.
std::mutex g_mutex;
LLT g_llt;
LDLT g_ldlt;
bool g_llt_analyzed = false;
bool g_ldlt_analyzed = false;
std::vector<Index> g_indptr;
std::vector<Index> g_indices;

bool same_pattern(const ConstRefVecL& indptr, const ConstRefVecL& indices) {
  if (g_indptr.size() != static_cast<std::size_t>(indptr.size()) ||
      g_indices.size() != static_cast<std::size_t>(indices.size()))
    return false;
  return std::memcmp(g_indptr.data(), indptr.data(), g_indptr.size() * sizeof(Index)) == 0 &&
         std::memcmp(g_indices.data(), indices.data(), g_indices.size() * sizeof(Index)) == 0;
}

void remember_pattern(const ConstRefVecL& indptr, const ConstRefVecL& indices) {
  if (same_pattern(indptr, indices)) return;
  g_indptr.assign(indptr.data(), indptr.data() + indptr.size());
  g_indices.assign(indices.data(), indices.data() + indices.size());
  g_llt_analyzed = false;
  g_ldlt_analyzed = false;
}

Eigen::Map<const SpMat> csc_view(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data,
                                 Index n) {
  if (indptr.size() != n + 1) throw std::invalid_argument("indptr must have n+1 entries");
  if (data.size() != indptr[n]) throw std::invalid_argument("data must have indptr[n] entries");
  return Eigen::Map<const SpMat>(n, n, indptr[n], indptr.data(), indices.data(), data.data());
}

template <class Solver>
bool factor_and_solve(Solver& solver, bool& analyzed, const Eigen::Map<const SpMat>& A, const ConstRefRow& b,
                      RowMatrixXd& x) {
  if (!analyzed) {
    solver.analyzePattern(A);
    analyzed = true;
  }
  solver.factorize(A);
  if (solver.info() != Eigen::Success) return false;
  Eigen::MatrixXd rhs = b;
  Eigen::MatrixXd sol = solver.solve(rhs);
  if (solver.info() != Eigen::Success || !sol.allFinite()) return false;
  x = sol;
  return true;
}

}  // namespace

SolveMethod sparse_solve_cholesky(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data,
                                  Index n, const ConstRefRow& b, RowMatrixXd& x) {
  if (b.rows() != n) throw std::invalid_argument("b must have n rows");
  const auto A = csc_view(indptr, indices, data, n);
  std::lock_guard<std::mutex> lock(g_mutex);
  remember_pattern(indptr, indices);
  if (factor_and_solve(g_llt, g_llt_analyzed, A, b, x)) return SolveMethod::llt;
  if (factor_and_solve(g_ldlt, g_ldlt_analyzed, A, b, x)) return SolveMethod::ldlt;
  return SolveMethod::failed;
}

bool sparse_solve_ldlt(const ConstRefVecL& indptr, const ConstRefVecL& indices, const ConstRefVec& data, Index n,
                       const ConstRefRow& b, RowMatrixXd& x) {
  if (b.rows() != n) throw std::invalid_argument("b must have n rows");
  const auto A = csc_view(indptr, indices, data, n);
  std::lock_guard<std::mutex> lock(g_mutex);
  remember_pattern(indptr, indices);
  return factor_and_solve(g_ldlt, g_ldlt_analyzed, A, b, x);
}

}  // namespace simkit
