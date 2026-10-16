// pybind11 bindings exposing the simkit C++ core as ``simkit._cpp``.
//
// The bindings deal with NumPy shapes so the Python wrappers stay one-liners:
// per-element stacks come in as (t, d, d) / (t, b, b) arrays and go out the
// same way.
#include "simkit/parallel_for.h"
#include "simkit/polar_svd.h"
#include "simkit/psd_project.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

namespace py = pybind11;
using simkit::Index;
using simkit::RowMatrixXd;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

// View a (t, d, d) or (t, k) array as a (t, k) row-major matrix.
Eigen::Map<const RowMatrixXd> stack_view(const Arr& a) {
  if (a.ndim() == 3) return {a.data(), a.shape(0), a.shape(1) * a.shape(2)};
  if (a.ndim() == 2) return {a.data(), a.shape(0), a.shape(1)};
  throw std::invalid_argument("expected a 2D or 3D array");
}

// Wrap an Eigen matrix (moved, no copy) as a NumPy array with the given shape.
py::array to_array(RowMatrixXd&& M, std::vector<py::ssize_t> shape) {
  py::array arr = py::cast(std::move(M));
  return arr.reshape(std::move(shape));
}

}  // namespace

PYBIND11_MODULE(_cpp, m) {
  m.doc() = "simkit C++ core: batched decompositions and PSD projection";

  m.def("num_threads", &simkit::num_threads, "Number of worker threads used by the C++ core.");
  m.def("set_num_threads", &simkit::set_num_threads, py::arg("n"),
        "Set the worker thread count (n <= 0 restores the default).");

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
}
