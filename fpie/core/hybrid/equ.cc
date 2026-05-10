#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <tuple>
#include <vector>

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
  // Black nodes first: (i+j) odd -> indices [1, n_mid)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      if ((i + j) % 2 == 1)
        maskbuf[i * m + j] = (arr(i, j) > 0) ? ++cnt : 0;
  n_mid = cnt + 1;
  // Red nodes second: (i+j) even -> indices [n_mid, N)
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
  memcpy(tmp, X, sizeof(float) * N * 3);

  offset[0] = 0;
  int extra = N % n_proc;
  for (int i = 0; i < n_proc; ++i)
    offset[i + 1] = offset[i] + N / n_proc + (i < extra);
}

void HybridEquSolver::sync() {
  MPI_Bcast(&N,     1,        MPI_INT,   0, MPI_COMM_WORLD);
  MPI_Bcast(&n_mid, 1,        MPI_INT,   0, MPI_COMM_WORLD);
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
  memcpy(tmp, X, sizeof(float) * N * 3);
}

// Reads neighbours from X (updated by the previous half-sweep),
// writes new value directly into X[i] in-place.
// Safe within one colour: no black node neighbours another black node,
// so there are no read-write conflicts within a half-sweep.
inline void HybridEquSolver::update_equation(int i) {
  int off3 = i*3, off4 = i*4;
  int id0=A[off4+0]*3, id1=A[off4+1]*3, id2=A[off4+2]*3, id3=A[off4+3]*3;
  X[off3+0] = (B[off3+0]+X[id0+0]+X[id1+0]+X[id2+0]+X[id3+0]) / 4;
  X[off3+1] = (B[off3+1]+X[id0+1]+X[id1+1]+X[id2+1]+X[id3+1]) / 4;
  X[off3+2] = (B[off3+2]+X[id0+2]+X[id1+2]+X[id2+2]+X[id3+2]) / 4;
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

// ---------------------------------------------------------------------------
// Sync the updated colour [lo, hi) across all ranks.
// Each rank contributes the intersection of its slice with [lo, hi).
// Called twice per sweep: after black half-sweep, after red half-sweep.
// ---------------------------------------------------------------------------
void HybridEquSolver::allgather_range(int lo, int hi) {
  std::vector<int> counts(n_proc, 0), displs(n_proc, 0);
  for (int r = 0; r < n_proc; ++r) {
    int a = std::max(offset[r],   lo);
    int b = std::min(offset[r+1], hi);
    counts[r] = (b > a) ? (b - a) * 3 : 0;
    displs[r] = a * 3;
  }
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                 X, counts.data(), displs.data(),
                 MPI_FLOAT, MPI_COMM_WORLD);
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
HybridEquSolver::step(int iteration, float eps) {
  int converged = 0;

  // This rank's slice of [0, N). Index 0 is boundary constant, skip it.
  int r_lo = offset[proc_id];
  int r_hi = offset[proc_id + 1];

  // Intersection with black [1, n_mid) and red [n_mid, N) halves.
  int blk_lo = std::max(r_lo, 1),     blk_hi = std::min(r_hi, n_mid);
  int red_lo = std::max(r_lo, n_mid), red_hi = r_hi;

  // -------------------------------------------------------------------------
  // Red/Black Gauss-Seidel with MPI + OpenMP
  //
  // Checkerboard property: every black node's 4 neighbours are all red, and
  // vice versa.  So within each half-sweep all updates are independent ->
  // safe to parallelise with OpenMP, no tmp buffer needed.
  //
  // Two MPI syncs per full sweep (one per colour) vs one sync per Jacobi
  // iteration.  But each sync transfers only half the data, and crucially,
  // convergence rate matches single-threaded Gauss-Seidel (gcc backend)
  // because each half-sweep immediately sees the other colour's latest values.
  // -------------------------------------------------------------------------
  for (int i = 0; i < iteration && !converged; ++i) {

    // --- Black half-sweep: update X[1..n_mid) on this rank's slice ----------
    if (blk_lo < blk_hi) {
#pragma omp parallel for schedule(static)
      for (int k = blk_lo; k < blk_hi; ++k)
        update_equation(k);
    }
    allgather_range(1, n_mid);   // all ranks now have fresh black values

    // --- Red half-sweep: update X[n_mid..N) on this rank's slice ------------
    if (red_lo < red_hi) {
#pragma omp parallel for schedule(static)
      for (int k = red_lo; k < red_hi; ++k)
        update_equation(k);
    }
    allgather_range(n_mid, N);   // all ranks now have fresh red values

    // --- Convergence check every min_interval full sweeps -------------------
    if (eps > 0.0f && (i + 1) % min_interval == 0) {
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
