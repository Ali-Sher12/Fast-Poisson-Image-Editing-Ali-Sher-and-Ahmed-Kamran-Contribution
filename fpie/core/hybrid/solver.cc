#include "solver.h"

PYBIND11_MODULE(core_hybrid, m) {
  py::class_<HybridEquSolver>(m, "EquSolver")
      .def(py::init<int, int>())        // n_cpu, min_interval
      .def("partition", &HybridEquSolver::partition)
      .def("reset",     &HybridEquSolver::reset)
      .def("sync",      &HybridEquSolver::sync)
      .def("step",      &HybridEquSolver::step,
           py::arg("iteration"), py::arg("eps") = 0.0f);

  py::class_<HybridGridSolver>(m, "GridSolver")
      .def(py::init<int, int, int, int>())  // grid_x, grid_y, n_cpu, min_interval
      .def("reset", &HybridGridSolver::reset)
      .def("sync",  &HybridGridSolver::sync)
      .def("step",  &HybridGridSolver::step,
           py::arg("iteration"), py::arg("eps") = 0.0f);
}
