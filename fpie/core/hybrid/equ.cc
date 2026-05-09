#include <mpi.h>
#include <omp.h>

#include <tuple>

#include "solver.h"

// How often (in iters) to check convergence inside the sync window.
static const int EPS_CHECK_INTERVAL = 100;

HybridEquSolver::HybridEquSolver(int n_cpu, int min_interval)
    : maskbuf(NULL),
      imgbuf(NULL),
      tmp(NULL),
      n_mid(0),
      min_interval(min_interval),
      EquSolver() {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  offset = new int[n_proc + 1];
  omp_set_num_threads(n_cpu);
}

HybridEquSolver::~HybridEquSolver() {
  if (maskbuf != NULL) { delete[] maskbuf; delete[] imgbuf; }
  if (tmp != NULL) { delete[] tmp; }
  delete[] offset;
}

// Red/black partition identical to OpenMP solver so pybind integration works.
py::array_t<int> HybridEquSolver::partition(py::array_t<int> mask) {
  auto arr = mask.unchecked<2>();
  int n = arr.shape(0), m = arr.shape(1);
  if (maskbuf != NULL) delete[] maskbuf;
  maskbuf = new int[n * m];
  int cnt = 0;
  // odd (red)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      if ((i + j) % 2 == 1)
        maskbuf[i * m + j] = (arr(i, j) > 0) ? ++cnt : 0;
  n_mid = cnt + 1;
  // even (black)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      if ((i + j) % 2 == 0)
        maskbuf[i * m + j] = (arr(i, j) > 0) ? ++cnt : 0;
  return py::array({n, m}, maskbuf);
}

void HybridEquSolver::post_reset() {
  if (tmp != NULL) { delete[] tmp; delete[] imgbuf; }
  tmp    = new float[N * 3];
  imgbuf = new unsigned char[N * 3];
  // Divide equations evenly — N already counts only mask pixels so this
  // is inherently mask-aware for EquSolver.
  offset[0] = 0;
  int extra = N % n_proc;
  for (int i = 0; i < n_proc; ++i)
    offset[i + 1] = offset[i] + N / n_proc + (i < extra);
}

void HybridEquSolver::sync() {
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (proc_id > 0) {
    if (A != NULL) { delete[] A; delete[] B; delete[] X; delete[] tmp; delete[] imgbuf; }
    A      = new int[N * 4];
    B      = new float[N * 3];
    X      = new float[N * 3];
    tmp    = new float[N * 3];
    imgbuf = new unsigned char[N * 3];
  }
  MPI_Bcast(A,      N * 4, MPI_INT,   0, MPI_COMM_WORLD);
  MPI_Bcast(B,      N * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(X,      N * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(offset, n_proc + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

inline void HybridEquSolver::update_equation(int i) {
  int off3 = i * 3, off4 = i * 4;
  int id0 = A[off4+0]*3, id1 = A[off4+1]*3, id2 = A[off4+2]*3, id3 = A[off4+3]*3;
  X[off3+0] = (B[off3+0] + X[id0+0] + X[id1+0] + X[id2+0] + X[id3+0]) / 4;
  X[off3+1] = (B[off3+1] + X[id0+1] + X[id1+1] + X[id2+1] + X[id3+1]) / 4;
  X[off3+2] = (B[off3+2] + X[id0+2] + X[id1+2] + X[id2+2] + X[id3+2]) / 4;
}

void HybridEquSolver::calc_error() {
  // Parallelise residual computation with OpenMP within this rank.
#pragma omp parallel for schedule(static)
  for (int i = 1; i < N; ++i) {
    int off3 = i*3, off4 = i*4;
    int id0=A[off4+0]*3, id1=A[off4+1]*3, id2=A[off4+2]*3, id3=A[off4+3]*3;
    tmp[off3+0]=std::abs(4*X[off3+0]-(X[id0+0]+X[id1+0]+X[id2+0]+X[id3+0])-B[off3+0]);
    tmp[off3+1]=std::abs(4*X[off3+1]-(X[id0+1]+X[id1+1]+X[id2+1]+X[id3+1])-B[off3+1]);
    tmp[off3+2]=std::abs(4*X[off3+2]-(X[id0+2]+X[id1+2]+X[id2+2]+X[id3+2])-B[off3+2]);
  }
  memset(err, 0, sizeof(err));
  for (int i = 1; i < N; ++i) {
    err[0] += tmp[i*3+0]; err[1] += tmp[i*3+1]; err[2] += tmp[i*3+2];
  }
}

bool HybridEquSolver::has_converged(float eps) {
  if (eps <= 0.0f || N <= 1) return false;
  calc_error();
  float mean = (err[0] + err[1] + err[2]) / (3.0f * (N - 1));
  return mean < eps;
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
HybridEquSolver::step(int iteration, float eps) {
  int converged = 0;

  for (int i = 0; i < iteration && !converged; i += min_interval) {
    // ---- Two-level Jacobi: MPI rank owns [offset[proc_id], offset[proc_id+1])
    //      OpenMP threads share the work inside that range. ----
    for (int s = 0; s < min_interval; ++s) {
      // Red half (indices < n_mid that belong to this rank)
      int r_start = std::max(offset[proc_id],   1);
      int r_end   = std::min(offset[proc_id+1], n_mid);
      int b_start = std::max(offset[proc_id],   n_mid);
      int b_end   = std::min(offset[proc_id+1], N);

#pragma omp parallel for schedule(static)
      for (int k = r_start; k < r_end; ++k) update_equation(k);

#pragma omp parallel for schedule(static)
      for (int k = b_start; k < b_end; ++k) update_equation(k);
    }

    // ---- MPI gather → broadcast (same protocol as original MPI solver) ----
    if (proc_id == 0) {
      for (int j = 1; j < n_proc; ++j)
        MPI_Recv(&X[offset[j]*3], (offset[j+1]-offset[j])*3,
                 MPI_FLOAT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Send(&X[offset[proc_id]*3], (offset[proc_id+1]-offset[proc_id])*3,
               MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(X, N * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ---- Adaptive convergence: rank 0 checks, broadcasts decision ----
    if (eps > 0.0f) {
      if (proc_id == 0) converged = has_converged(eps) ? 1 : 0;
      MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  if (proc_id == 0) {
    if (eps <= 0.0f) calc_error();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N * 3; ++i)
      imgbuf[i] = X[i] < 0 ? 0 : X[i] > 255 ? 255 : X[i];
    return std::make_tuple(py::array({N, 3}, imgbuf), py::array(3, err));
  } else {
    return std::make_tuple(py::array({1, 3}, imgbuf), py::array(3, err));
  }
}
