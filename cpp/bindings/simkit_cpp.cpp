// pybind11 bindings exposing the simkit C++ core as ``simkit._cpp``.
//
// The bindings deal with NumPy shapes so the Python wrappers stay one-liners:
// per-element stacks come in as (t, d, d) / (t, b, b) arrays and go out the
// same way; material parameters accept (t, 1), (t,) or scalar arrays.
#include "simkit/block_diag_sandwich.h"
#include "simkit/deformation_jacobian.h"
#include "simkit/energies/elastic.h"
#include "simkit/parallel_for.h"
#include "simkit/polar_svd.h"
#include "simkit/psd_project.h"
#include "simkit/rotation_gradient.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace py = pybind11;
using simkit::Index;
using simkit::RowMatrixXd;
using simkit::VectorXl;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrL = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

// View a (t, d, d) or (t, k) array as a (t, k) row-major matrix.
Eigen::Map<const RowMatrixXd> stack_view(const Arr& a) {
  if (a.ndim() == 3) return {a.data(), a.shape(0), a.shape(1) * a.shape(2)};
  if (a.ndim() == 2) return {a.data(), a.shape(0), a.shape(1)};
  throw std::invalid_argument("expected a 2D or 3D array");
}

Eigen::Map<const Eigen::VectorXd> vec_view(const Arr& a) { return {a.data(), static_cast<Index>(a.size())}; }
Eigen::Map<const VectorXl> lvec_view(const ArrL& a) { return {a.data(), static_cast<Index>(a.size())}; }

Eigen::Map<const simkit::RowMatrixXl> lmat_view(const ArrL& a) {
  if (a.ndim() != 2) throw std::invalid_argument("expected a 2D integer array");
  return {a.data(), a.shape(0), a.shape(1)};
}

// Wrap an Eigen matrix (moved, no copy) as a NumPy array with the given shape.
py::array to_array(RowMatrixXd&& M, std::vector<py::ssize_t> shape) {
  py::array arr = py::cast(std::move(M));
  return arr.reshape(std::move(shape));
}
py::array to_array(Eigen::VectorXd&& v, std::vector<py::ssize_t> shape) {
  py::array arr = py::cast(std::move(v));
  return arr.reshape(std::move(shape));
}
py::array to_array(VectorXl&& v) { return py::cast(std::move(v)); }


// Energy (t, 1), gradient (t, d, d), Hessian (t, d*d, d*d) wrappers.
template <class Fn>
py::array energy_wrap(const Arr& F, Fn fn) {
  const auto Fv = stack_view(F);
  Eigen::VectorXd out;
  {
    py::gil_scoped_release release;
    out = fn(Fv);
  }
  return to_array(std::move(out), {Fv.rows(), 1});
}

template <class Fn>
py::array gradient_wrap(const Arr& F, Fn fn) {
  const auto Fv = stack_view(F);
  const int d = simkit::dim_from_cols(Fv.cols());
  RowMatrixXd out;
  {
    py::gil_scoped_release release;
    out = fn(Fv);
  }
  return to_array(std::move(out), {Fv.rows(), d, d});
}

template <class Fn>
py::array hessian_wrap(const Arr& F, Fn fn) {
  const auto Fv = stack_view(F);
  const int d = simkit::dim_from_cols(Fv.cols());
  RowMatrixXd out;
  {
    py::gil_scoped_release release;
    out = fn(Fv);
  }
  return to_array(std::move(out), {Fv.rows(), d * d, d * d});
}

}  // namespace

PYBIND11_MODULE(_cpp, m) {
  m.doc() = "simkit C++ core: per-element energies, decompositions and assembly";

  m.def("num_threads", &simkit::num_threads, "Number of worker threads used by the C++ core.");
  m.def("set_num_threads", &simkit::set_num_threads, py::arg("n"),
        "Set the worker thread count (n <= 0 restores the default).");

  // ---- decompositions -----------------------------------------------------
  m.def(
      "svd_rv",
      [](const Arr& F) {
        const auto Fv = stack_view(F);
        const int d = simkit::dim_from_cols(Fv.cols());
        RowMatrixXd U, S, V;
        {
          py::gil_scoped_release release;
          simkit::svd_rv(Fv, U, S, V);
        }
        const Index t = Fv.rows();
        return py::make_tuple(to_array(std::move(U), {t, d, d}), to_array(std::move(S), {t, d, d}),
                              to_array(std::move(V), {t, d, d}));
      },
      py::arg("F"), "Rotation-variant SVD of a (t, d, d) stack: (U, S, V) with S as diagonal matrices.");

  m.def(
      "polar_svd",
      [](const Arr& F) {
        const auto Fv = stack_view(F);
        const int d = simkit::dim_from_cols(Fv.cols());
        RowMatrixXd R, S;
        {
          py::gil_scoped_release release;
          simkit::polar_svd(Fv, R, S);
        }
        const Index t = Fv.rows();
        return py::make_tuple(to_array(std::move(R), {t, d, d}), to_array(std::move(S), {t, d, d}));
      },
      py::arg("F"), "Polar decomposition F = R S of a (t, d, d) stack.");

  m.def(
      "rotation_gradient_F",
      [](const Arr& F) { return hessian_wrap(F, [](const auto& Fv) { return simkit::rotation_gradient_F(Fv); }); },
      py::arg("F"), "dR/dF of the polar rotation, (t, d*d, d*d).");

  m.def(
      "psd_project",
      [](const Arr& H, const std::string& method) {
        const auto Hv = stack_view(H);
        const int b = simkit::block_from_cols(Hv.cols());
        RowMatrixXd out;
        {
          py::gil_scoped_release release;
          out = simkit::psd_project(Hv, method);
        }
        return to_array(std::move(out), {Hv.rows(), b, b});
      },
      py::arg("H"), py::arg("method") = "proj", "Project a (t, b, b) stack of symmetric blocks to PSD.");

  // ---- materials ------------------------------------------------------------
  m.def("arap_energy_element_F",
        [](const Arr& F, const Arr& mu) {
          return energy_wrap(F, [&](const auto& Fv) { return simkit::arap_energy_element_F(Fv, vec_view(mu)); });
        },
        py::arg("F"), py::arg("mu"));
  m.def("arap_gradient_element_F",
        [](const Arr& F, const Arr& mu) {
          return gradient_wrap(F, [&](const auto& Fv) { return simkit::arap_gradient_element_F(Fv, vec_view(mu)); });
        },
        py::arg("F"), py::arg("mu"));
  m.def("arap_hessian_element_F",
        [](const Arr& F, const Arr& mu) {
          return hessian_wrap(F, [&](const auto& Fv) { return simkit::arap_hessian_element_F(Fv, vec_view(mu)); });
        },
        py::arg("F"), py::arg("mu"));

#define SIMKIT_BIND_MU_LAM(NAME)                                                                                   \
  m.def(#NAME "_energy_element_F",                                                                                 \
        [](const Arr& F, const Arr& mu, const Arr& lam) {                                                          \
          return energy_wrap(F, [&](const auto& Fv) { return simkit::NAME##_energy_element_F(Fv, vec_view(mu), vec_view(lam)); }); \
        },                                                                                                         \
        py::arg("F"), py::arg("mu"), py::arg("lam"));                                                              \
  m.def(#NAME "_gradient_element_F",                                                                               \
        [](const Arr& F, const Arr& mu, const Arr& lam) {                                                          \
          return gradient_wrap(F, [&](const auto& Fv) { return simkit::NAME##_gradient_element_F(Fv, vec_view(mu), vec_view(lam)); }); \
        },                                                                                                         \
        py::arg("F"), py::arg("mu"), py::arg("lam"));                                                              \
  m.def(#NAME "_hessian_element_F",                                                                                \
        [](const Arr& F, const Arr& mu, const Arr& lam) {                                                          \
          return hessian_wrap(F, [&](const auto& Fv) { return simkit::NAME##_hessian_element_F(Fv, vec_view(mu), vec_view(lam)); }); \
        },                                                                                                         \
        py::arg("F"), py::arg("mu"), py::arg("lam"));

  SIMKIT_BIND_MU_LAM(fcr)
  SIMKIT_BIND_MU_LAM(neo_hookean)
  SIMKIT_BIND_MU_LAM(stable_neo_hookean)
  SIMKIT_BIND_MU_LAM(stvk)
  SIMKIT_BIND_MU_LAM(linear_elasticity)
  SIMKIT_BIND_MU_LAM(macklin_mueller_neo_hookean)
#undef SIMKIT_BIND_MU_LAM

  m.def(
      "elastic_energy_element_F",
      [](const Arr& F, const Arr& mu, const Arr& lam, const std::string& material) {
        const auto mat = simkit::material_from_string(material);
        return energy_wrap(F, [&](const auto& Fv) { return simkit::elastic_energy_element_F(Fv, vec_view(mu), vec_view(lam), mat); });
      },
      py::arg("F"), py::arg("mu"), py::arg("lam"), py::arg("material"));
  m.def(
      "elastic_gradient_element_F",
      [](const Arr& F, const Arr& mu, const Arr& lam, const std::string& material) {
        const auto mat = simkit::material_from_string(material);
        return gradient_wrap(F, [&](const auto& Fv) { return simkit::elastic_gradient_element_F(Fv, vec_view(mu), vec_view(lam), mat); });
      },
      py::arg("F"), py::arg("mu"), py::arg("lam"), py::arg("material"));
  m.def(
      "elastic_hessian_element_F",
      [](const Arr& F, const Arr& mu, const Arr& lam, const std::string& material, bool psd) {
        const auto mat = simkit::material_from_string(material);
        return hessian_wrap(F, [&](const auto& Fv) { return simkit::elastic_hessian_element_F(Fv, vec_view(mu), vec_view(lam), mat, psd); });
      },
      py::arg("F"), py::arg("mu"), py::arg("lam"), py::arg("material"), py::arg("psd") = true);

  // ---- operators / assembly -------------------------------------------------
  m.def(
      "deformation_jacobian_coo",
      [](const Arr& X, const ArrL& T) {
        if (X.ndim() != 2) throw std::invalid_argument("X must be (n, dim)");
        const auto Xv = stack_view(X);
        const auto Tv = lmat_view(T);
        VectorXl rows, cols;
        Eigen::VectorXd vals;
        {
          py::gil_scoped_release release;
          simkit::deformation_jacobian_triplets(Xv, Tv, rows, cols, vals);
        }
        return py::make_tuple(to_array(std::move(rows)), to_array(std::move(cols)), py::cast(std::move(vals)));
      },
      py::arg("X"), py::arg("T"), "COO triplets (rows, cols, vals) of the deformation Jacobian.");

  m.def(
      "block_diag_sandwich_csr",
      [](const ArrL& indptr, const ArrL& indices, const Arr& data, Index ncols, const Arr& He) {
        const auto Hv = stack_view(He);
        VectorXl out_indptr, out_indices;
        Eigen::VectorXd out_data;
        {
          py::gil_scoped_release release;
          simkit::block_diag_sandwich_csr(lvec_view(indptr), lvec_view(indices), vec_view(data), ncols, Hv, out_indptr,
                                          out_indices, out_data);
        }
        return py::make_tuple(to_array(std::move(out_indptr)), to_array(std::move(out_indices)),
                              py::cast(std::move(out_data)));
      },
      py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("ncols"), py::arg("He"),
      "CSR (indptr, indices, data) of J^T blockdiag(He) J for a CSR J.");

  m.def(
      "block_diag_sandwich_dense",
      [](const Arr& J, const Arr& He) {
        if (J.ndim() != 2) throw std::invalid_argument("J must be 2D");
        const auto Jv = stack_view(J);
        const auto Hv = stack_view(He);
        RowMatrixXd out;
        {
          py::gil_scoped_release release;
          out = simkit::block_diag_sandwich_dense(Jv, Hv);
        }
        return to_array(std::move(out), {out.rows(), out.cols()});
      },
      py::arg("J"), py::arg("He"), "Dense J^T blockdiag(He) J for a dense J.");
}
