#include <mpi.h>
#include <omp.h>

#include <tuple>

#include "solver.h"

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

py::array_t<int> HybridEquSolver::partition(py::array_t<int> mask) {
  auto arr = mask.unchecked<2>();
  int n = arr.shape(0), m = arr.shape(1);
  if (maskbuf != NULL) delete[] maskbuf;
  maskbuf = new int[n * m];
  int cnt = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      if ((i + j) % 2 == 1)
        maskbuf[i * m + j] = (arr(i, j) > 0) ? ++cnt : 0;
  n_mid = cnt + 1;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      if ((i + j) % 2 == 0)
        maskbuf[i * m + j] = (arr(i, j) > 0) ? ++cnt : 0;
  return py::array({n, m}, maskbuf);
}

void HybridEquSolver::post_reset() {
  if (tmp != NULL) { delete[] tmp; delete[] imgbuf; }
  // tmp is used as X_new (double buffer) to avoid race conditions.
  tmp    = new float[N * 3];
  imgbuf = new unsigned char[N * 3];
  memcpy(tmp, X, sizeof(float) * N * 3);

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
  MPI_Bcast(A,      N * 4,    MPI_INT,   0, MPI_COMM_WORLD);
  MPI_Bcast(B,      N * 3,    MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(X,      N * 3,    MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(offset, n_proc+1, MPI_INT,   0, MPI_COMM_WORLD);
  // Initialise tmp to match X on all ranks.
  memcpy(tmp, X, sizeof(float) * N * 3);
}

// Reads from X (old values), writes result to tmp[i] (new values).
// No race condition: all threads read X simultaneously, write to
// disjoint tmp[i] locations.
inline void HybridEquSolver::update_equation(int i) {
  int off3 = i*3, off4 = i*4;
  int id0=A[off4+0]*3, id1=A[off4+1]*3, id2=A[off4+2]*3, id3=A[off4+3]*3;
  tmp[off3+0] = (B[off3+0]+X[id0+0]+X[id1+0]+X[id2+0]+X[id3+0]) / 4;
  tmp[off3+1] = (B[off3+1]+X[id0+1]+X[id1+1]+X[id2+1]+X[id3+1]) / 4;
  tmp[off3+2] = (B[off3+2]+X[id0+2]+X[id1+2]+X[id2+2]+X[id3+2]) / 4;
}

void HybridEquSolver::calc_error() {
  memset(err, 0, sizeof(err));
  for (int i = 1; i < N; ++i) {
    int off3=i*3, off4=i*4;
    int id0=A[off4+0]*3, id1=A[off4+1]*3, id2=A[off4+2]*3, id3=A[off4+3]*3;
    err[0]+=std::abs(4*X[off3+0]-(X[id0+0]+X[id1+0]+X[id2+0]+X[id3+0])-B[off3+0]);
    err[1]+=std::abs(4*X[off3+1]-(X[id0+1]+X[id1+1]+X[id2+1]+X[id3+1])-B[off3+1]);
    err[2]+=std::abs(4*X[off3+2]-(X[id0+2]+X[id1+2]+X[id2+2]+X[id3+2])-B[off3+2]);
  }
}

bool HybridEquSolver::has_converged(float eps) {
  if (eps <= 0.0f || N <= 1) return false;
  calc_error();
  return (err[0]+err[1]+err[2]) / (3.0f*(N-1)) < eps;
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
HybridEquSolver::step(int iteration, float eps) {
  int converged = 0;

  // k range for this rank (skip index 0 which is the boundary constant).
  int k_start = std::max(offset[proc_id],   1);
  int k_end   =         offset[proc_id+1];

  for (int i = 0; i < iteration && !converged; i += min_interval) {
    for (int s = 0; s < min_interval; ++s) {
      // Step 1: each thread writes updated value to tmp[k] reading from X.
      //         No race: reads from X (shared-read), writes to disjoint tmp[k].
#pragma omp parallel for schedule(static)
      for (int k = k_start; k < k_end; ++k)
        update_equation(k);

      // Step 2: copy this rank's updated slice from tmp back into X.
#pragma omp parallel for schedule(static)
      for (int k = k_start; k < k_end; ++k) {
        X[k*3+0] = tmp[k*3+0];
        X[k*3+1] = tmp[k*3+1];
        X[k*3+2] = tmp[k*3+2];
      }
    }

    // Step 3: gather all ranks' slices into rank 0, then broadcast full X.
    if (proc_id == 0) {
      for (int j = 1; j < n_proc; ++j)
        MPI_Recv(&X[offset[j]*3], (offset[j+1]-offset[j])*3,
                 MPI_FLOAT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Send(&X[k_start*3], (k_end - k_start)*3,
               MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(X, N * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Keep tmp in sync so next iteration reads correct old values.
    memcpy(tmp, X, sizeof(float) * N * 3);

    // Adaptive convergence.
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
