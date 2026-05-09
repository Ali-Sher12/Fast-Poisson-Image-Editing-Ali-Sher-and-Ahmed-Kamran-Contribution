#ifndef FPIE_CORE_HYBRID_SOLVER_H_
#define FPIE_CORE_HYBRID_SOLVER_H_

#include <tuple>
#include "base_solver.h"

// ---------------------------------------------------------------------------
// HybridEquSolver
// MPI distributes equations across ranks (coarse-grained).
// OpenMP threads parallelize the Jacobi update within each rank (fine-grained).
// Adaptive convergence: stops early when mean residual < eps.
// ---------------------------------------------------------------------------
class HybridEquSolver : public EquSolver {
  int* maskbuf;
  unsigned char* imgbuf;
  float* tmp;
  int n_mid;          // red/black split boundary (same as OpenMP)
  int proc_id;        // MPI rank
  int n_proc;         // total MPI ranks
  int* offset;        // offset[i]..offset[i+1] = equation range for rank i
  int min_interval;   // MPI sync every min_interval iterations

 public:
  explicit HybridEquSolver(int n_cpu, int min_interval);
  ~HybridEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();
  void sync();

  inline void update_equation(int i);
  void calc_error();
  bool has_converged(float eps);

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration, float eps = 0.0f);
};

// ---------------------------------------------------------------------------
// HybridGridSolver
// MPI distributes image rows across ranks with MASK-AWARE load balancing.
// OpenMP threads parallelize pixel updates within each rank's rows.
// Adaptive convergence: stops early when mean residual < eps.
// ---------------------------------------------------------------------------
class HybridGridSolver : public GridSolver {
  unsigned char* imgbuf;
  int proc_id;
  int n_proc;
  int* offset;        // mask-aware row partition
  int min_interval;

 public:
  explicit HybridGridSolver(int grid_x, int grid_y, int n_cpu, int min_interval);
  ~HybridGridSolver();

  void post_reset();
  void sync();

  inline void update_equation(int id);
  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration, float eps = 0.0f);
};

#endif  // FPIE_CORE_HYBRID_SOLVER_H_
