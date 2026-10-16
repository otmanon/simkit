// Self-contained checks for the simkit C++ core (no test framework needed).
#include "simkit/polar_svd.h"
#include "simkit/psd_project.h"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

using namespace simkit;

static int g_failures = 0;
#define CHECK(cond, msg)                                                             \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, std::string(msg).c_str()); \
      ++g_failures;                                                                  \
    }                                                                                \
  } while (0)

static RowMatrixXd random_F(std::mt19937& rng, Index t, int dim, double scale = 0.1) {
  std::normal_distribution<double> N(0.0, 1.0);
  RowMatrixXd F(t, dim * dim);
  for (Index e = 0; e < t; ++e)
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) F(e, i * dim + j) = (i == j ? 1.0 : 0.0) + scale * N(rng);
  return F;
}

static void test_decompositions(std::mt19937& rng) {
  for (int dim : {2, 3}) {
    RowMatrixXd F = random_F(rng, 8, dim, 0.3);
    F.row(1) *= -1.0;  // one reflected element
    RowMatrixXd R, S, U, Sd, V;
    polar_svd(F, R, S);
    svd_rv(F, U, Sd, V);
    for (Index e = 0; e < F.rows(); ++e) {
      Eigen::Map<const RowMatrixXd> Fe(F.row(e).data(), dim, dim), Re(R.row(e).data(), dim, dim),
          Se(S.row(e).data(), dim, dim), Ue(U.row(e).data(), dim, dim), Ve(V.row(e).data(), dim, dim),
          Sde(Sd.row(e).data(), dim, dim);
      CHECK((Re * Se - Fe).norm() < 1e-10, "polar reconstruction");
      CHECK(std::abs(Re.determinant() - 1.0) < 1e-10, "R is a rotation");
      CHECK((Se - Se.transpose()).norm() < 1e-10, "S symmetric");
      CHECK((Ue * Sde * Ve.transpose() - Fe).norm() < 1e-10, "svd reconstruction");
      // The sign fix only acts when exactly one factor is a reflection, so
      // U V^T is always a rotation (both may still be reflections together).
      CHECK((Ue * Ve.transpose()).determinant() > 0, "svd_rv: U V^T is a rotation");
    }
  }
}

static void test_psd(std::mt19937& rng) {
  std::normal_distribution<double> N(0.0, 1.0);
  for (int b : {3, 4, 9, 5}) {
    const Index t = 4;
    RowMatrixXd H(t, b * b);
    for (Index e = 0; e < t; ++e) {
      Eigen::MatrixXd A(b, b);
      for (int i = 0; i < b * b; ++i) A.data()[i] = N(rng);
      Eigen::Map<RowMatrixXd>(H.row(e).data(), b, b) = A + A.transpose();
    }
    RowMatrixXd P = psd_project(H, "proj");
    RowMatrixXd Pabs = psd_project(H, "abs");
    for (Index e = 0; e < t; ++e) {
      Eigen::Map<const RowMatrixXd> Pe(P.row(e).data(), b, b), Pa(Pabs.row(e).data(), b, b);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Pe), ea(Pa);
      CHECK(es.eigenvalues().minCoeff() >= 1e-6 - 1e-12, "psd_project floors eigenvalues");
      CHECK(ea.eigenvalues().minCoeff() >= -1e-12, "psd_project abs is PSD");
    }
  }
}

int main() {
  std::mt19937 rng(0);
  test_decompositions(rng);
  test_psd(rng);
  if (g_failures == 0) std::printf("simkit C++ tests: all passed\n");
  return g_failures == 0 ? 0 : 1;
}
