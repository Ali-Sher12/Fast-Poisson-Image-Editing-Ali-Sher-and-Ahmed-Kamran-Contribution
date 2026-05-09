#ifndef FPIE_CORE_OPENMP_SOLVER_H_
#define FPIE_CORE_OPENMP_SOLVER_H_

#include <tuple>
#include "base_solver.h"

class OpenMPEquSolver : public EquSolver {
  int* maskbuf;
  unsigned char* imgbuf;
  float* tmp;
  int n_mid;

 public:
  explicit OpenMPEquSolver(int n_cpu);
  ~OpenMPEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();
  inline void update_equation(int i);
  void calc_error();
  bool has_converged(float eps);

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration, float eps = 0.0f);
};

class OpenMPGridSolver : public GridSolver {
  unsigned char* imgbuf;
  float* tmp;
  int n_mask_pixels;

 public:
  OpenMPGridSolver(int grid_x, int grid_y, int n_cpu);
  ~OpenMPGridSolver();

  void post_reset();
  inline void update_equation(int id);
  void calc_error();
  bool has_converged(float eps);

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration, float eps = 0.0f);
};

#endif  // FPIE_CORE_OPENMP_SOLVER_H_
