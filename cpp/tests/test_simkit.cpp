// Self-contained checks for the simkit C++ core (no test framework needed).
// Finite-difference checks of every material gradient and Hessian, plus the
// decompositions and the assembly helpers.
#include "simkit/block_diag_sandwich.h"
#include "simkit/deformation_jacobian.h"
#include "simkit/energies/elastic.h"
#include "simkit/polar_svd.h"
#include "simkit/psd_project.h"
#include "simkit/rotation_gradient.h"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace simkit;

static int g_failures = 0;
#define CHECK(cond, msg)                                                                 \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, std::string(msg).c_str());     \
      ++g_failures;                                                                      \
    }                                                                                    \
  } while (0)

static RowMatrixXd random_F(std::mt19937& rng, Index t, int dim, double scale = 0.1) {
  std::normal_distribution<double> N(0.0, 1.0);
  RowMatrixXd F(t, dim * dim);
  for (Index e = 0; e < t; ++e)
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) F(e, i * dim + j) = (i == j ? 1.0 : 0.0) + scale * N(rng);
  return F;
}

static void test_material(const std::string& name, std::mt19937& rng) {
  const Material mat = material_from_string(name);
  for (int dim : {2, 3}) {
    const Index t = 6;
    const int b = dim * dim;
    RowMatrixXd F = random_F(rng, t, dim);
    Eigen::VectorXd mu = Eigen::VectorXd::Constant(t, 1.3), lam = Eigen::VectorXd::Constant(t, 2.1);
    Eigen::VectorXd psi = elastic_energy_element_F(F, mu, lam, mat);
    RowMatrixXd P = elastic_gradient_element_F(F, mu, lam, mat);
    RowMatrixXd H = elastic_hessian_element_F(F, mu, lam, mat, false);
    const double h = 1e-6;
    for (Index e = 0; e < t; ++e)
      for (int k = 0; k < b; ++k) {
        RowMatrixXd Fp = F, Fm = F;
        Fp(e, k) += h;
        Fm(e, k) -= h;
        const double dpsi = (elastic_energy_element_F(Fp, mu, lam, mat)[e] - elastic_energy_element_F(Fm, mu, lam, mat)[e]) / (2 * h);
        CHECK(std::abs(dpsi - P(e, k)) < 1e-5, name + " gradient dim=" + std::to_string(dim));
        RowMatrixXd Pp = elastic_gradient_element_F(Fp, mu, lam, mat), Pm = elastic_gradient_element_F(Fm, mu, lam, mat);
        for (int l = 0; l < b; ++l) {
          const double dP = (Pp(e, l) - Pm(e, l)) / (2 * h);
          CHECK(std::abs(dP - H(e, l * b + k)) < 1e-4, name + " hessian dim=" + std::to_string(dim));
        }
      }
    // Rest state has zero energy and stress.
    RowMatrixXd I(1, b);
    Eigen::Map<RowMatrixXd>(I.data(), dim, dim).setIdentity();
    // Stable Neo-Hookean is offset at rest by construction; the others vanish.
    if (mat != Material::stable_neo_hookean)
      CHECK(std::abs(elastic_energy_element_F(I, mu.head(1), lam.head(1), mat)[0]) < 1e-12, name + " rest energy");
    CHECK(elastic_gradient_element_F(I, mu.head(1), lam.head(1), mat).norm() < 1e-12, name + " rest stress");
  }
}

static void test_decompositions(std::mt19937& rng) {
  for (int dim : {2, 3}) {
    RowMatrixXd F = random_F(rng, 8, dim, 0.3);
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
    // dR/dF against finite differences.
    RowMatrixXd K = rotation_gradient_F(F);
    const int b = dim * dim;
    const double h = 1e-6;
    for (int k = 0; k < b; ++k) {
      RowMatrixXd Fp = F, Fm = F;
      Fp.col(k).array() += h;
      Fm.col(k).array() -= h;
      RowMatrixXd Rp, Sp, Rm, Sm;
      polar_svd(Fp, Rp, Sp);
      polar_svd(Fm, Rm, Sm);
      RowMatrixXd dR = (Rp - Rm) / (2 * h);
      for (Index e = 0; e < F.rows(); ++e)
        for (int l = 0; l < b; ++l) CHECK(std::abs(dR(e, l) - K(e, l * b + k)) < 1e-5, "rotation gradient");
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
    for (Index e = 0; e < t; ++e) {
      Eigen::Map<const RowMatrixXd> Pe(P.row(e).data(), b, b);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Pe);
      CHECK(es.eigenvalues().minCoeff() >= 1e-6 - 1e-12, "psd_project floors eigenvalues");
    }
  }
}

static void test_assembly(std::mt19937& rng) {
  // One tetrahedron: J maps rest positions to the identity deformation gradient.
  RowMatrixXd X(4, 3);
  X << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
  RowMatrixXl T(1, 4);
  T << 0, 1, 2, 3;
  Eigen::SparseMatrix<double> J = deformation_jacobian(X, T);
  Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(X.data(), 12);
  Eigen::VectorXd f = J * x;
  Eigen::Map<RowMatrixXd> F(f.data(), 3, 3);
  CHECK((F - RowMatrixXd::Identity(3, 3)).norm() < 1e-12, "deformation jacobian at rest");

  // J^T blockdiag(He) J against a dense computation.
  std::normal_distribution<double> N(0.0, 1.0);
  RowMatrixXd He(1, 81);
  Eigen::MatrixXd A(9, 9);
  for (int i = 0; i < 81; ++i) A.data()[i] = N(rng);
  Eigen::Map<RowMatrixXd>(He.data(), 9, 9) = A + A.transpose();
  Eigen::SparseMatrix<double, Eigen::RowMajor> Jr = J;
  Eigen::SparseMatrix<double> Q = block_diag_sandwich(Jr, He);
  Eigen::MatrixXd Jd = J;
  Eigen::MatrixXd Qd = Jd.transpose() * Eigen::Map<RowMatrixXd>(He.data(), 9, 9) * Jd;
  CHECK((Eigen::MatrixXd(Q) - Qd).norm() < 1e-10, "sparse block_diag_sandwich");
  RowMatrixXd Jdr = Jd;
  CHECK((block_diag_sandwich_dense(Jdr, He) - Qd).norm() < 1e-10, "dense block_diag_sandwich");
}

int main() {
  std::mt19937 rng(0);
  for (const char* name : {"arap", "fcr", "neo-hookean", "stable-neo-hookean", "stvk", "linear-elasticity",
                           "macklin-mueller-neo-hookean"})
    test_material(name, rng);
  test_decompositions(rng);
  test_psd(rng);
  test_assembly(rng);
  if (g_failures == 0) std::printf("simkit C++ tests: all passed\n");
  return g_failures == 0 ? 0 : 1;
}
