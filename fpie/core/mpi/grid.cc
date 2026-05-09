#include <mpi.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "solver.h"

MPIGridSolver::MPIGridSolver(int min_interval)
    : imgbuf(NULL), min_interval(min_interval), GridSolver(0, 0) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  offset = new int[n_proc + 1];
}

MPIGridSolver::~MPIGridSolver() {
  if (imgbuf != NULL) {
    delete[] imgbuf;
  }
  delete[] offset;
}

// ---------------------------------------------------------------------------
// MASK-AWARE LOAD BALANCING
// ---------------------------------------------------------------------------
// Original: rows divided equally -> load imbalance for irregular masks.
// New: count mask pixels per row, assign rows greedily so each rank gets
//      ~equal total mask pixels.  Row order is preserved for halo exchange.
// ---------------------------------------------------------------------------
void MPIGridSolver::post_reset() {
  if (imgbuf != NULL) {
    delete[] imgbuf;
  }
  imgbuf = new unsigned char[N * m3];

  std::vector<int> row_pixels(N, 0);
  int n_mask_pixels = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (mask[i * M + j]) {
        ++row_pixels[i];
        ++n_mask_pixels;
      }
    }
  }

  offset[0] = 0;

  if (n_mask_pixels == 0) {
    // Degenerate: no mask pixels, fall back to equal row split.
    int additional = N % n_proc;
    for (int i = 0; i < n_proc; ++i) {
      offset[i + 1] = offset[i] + N / n_proc + (i < additional);
    }
    return;
  }

  // Greedy scan: accumulate row pixel counts and cut when we hit each target.
  int target = (n_mask_pixels + n_proc - 1) / n_proc;
  int cumsum = 0, rank = 1;
  for (int i = 0; i < N && rank < n_proc; ++i) {
    cumsum += row_pixels[i];
    if (cumsum >= target * rank) {
      offset[rank++] = i + 1;
    }
  }
  while (rank <= n_proc) offset[rank++] = N;
}

void MPIGridSolver::sync() {
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (proc_id > 0) {
    if (mask != NULL) {
      delete[] mask;
      delete[] tgt;
      delete[] grad;
      delete[] imgbuf;
    }
    m3 = M * 3;
    mask = new int[N * M];
    tgt = new float[N * m3];
    grad = new float[N * m3];
    imgbuf = new unsigned char[N * m3];
  }
  MPI_Bcast(mask, N * M, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(tgt, N * m3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(grad, N * m3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(offset, n_proc + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

inline void MPIGridSolver::update_equation(int id) {
  int off3 = id * 3;
  int id0 = off3 - m3;
  int id1 = off3 - 3;
  int id2 = off3 + 3;
  int id3 = off3 + m3;
  tgt[off3 + 0] = (grad[off3+0] + tgt[id0+0] + tgt[id1+0] + tgt[id2+0] + tgt[id3+0]) / 4.0;
  tgt[off3 + 1] = (grad[off3+1] + tgt[id0+1] + tgt[id1+1] + tgt[id2+1] + tgt[id3+1]) / 4.0;
  tgt[off3 + 2] = (grad[off3+2] + tgt[id0+2] + tgt[id1+2] + tgt[id2+2] + tgt[id3+2]) / 4.0;
}

void MPIGridSolver::calc_error() {
  memset(err, 0, sizeof(err));
  for (int id = offset[proc_id] * M; id < offset[proc_id + 1] * M; ++id) {
    if (mask[id]) {
      int off3 = id * 3;
      int id0 = off3 - m3, id1 = off3 - 3, id2 = off3 + 3, id3 = off3 + m3;
      err[0] += std::abs(grad[off3+0]+tgt[id0+0]+tgt[id1+0]+tgt[id2+0]+tgt[id3+0]-tgt[off3+0]*4.0);
      err[1] += std::abs(grad[off3+1]+tgt[id0+1]+tgt[id1+1]+tgt[id2+1]+tgt[id3+1]-tgt[off3+1]*4.0);
      err[2] += std::abs(grad[off3+2]+tgt[id0+2]+tgt[id1+2]+tgt[id2+2]+tgt[id3+2]-tgt[off3+2]*4.0);
    }
  }
  if (proc_id == 0) {
    float tmp[3];
    for (int j = 1; j < n_proc; ++j) {
      MPI_Recv(&tgt[offset[j] * m3], (offset[j+1]-offset[j]) * m3, MPI_FLOAT,
               j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&tmp, 3, MPI_FLOAT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      err[0] += tmp[0]; err[1] += tmp[1]; err[2] += tmp[2];
    }
  } else {
    MPI_Send(&tgt[offset[proc_id]*m3], (offset[proc_id+1]-offset[proc_id])*m3,
             MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(&err, 3, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>> MPIGridSolver::step(
    int iteration, float eps) {
  int converged = 0;

  // Pre-compute global mask count once for residual normalisation.
  int local_mask_count = 0;
  for (int id = offset[proc_id]*M; id < offset[proc_id+1]*M; ++id)
    if (mask[id]) ++local_mask_count;
  int global_mask_count = 0;
  MPI_Allreduce(&local_mask_count, &global_mask_count, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  for (int i = 0; i < iteration && !converged; i += min_interval) {
    // --- Jacobi steps on this rank's row slice ---
    for (int j = 0; j < min_interval; ++j) {
      for (int k = offset[proc_id]*M; k < offset[proc_id+1]*M; ++k) {
        if (mask[k]) update_equation(k);
      }
    }

    // --- Halo exchange: share boundary rows with neighbours ---
    if (proc_id != n_proc - 1)
      MPI_Send(&tgt[(offset[proc_id+1]-1)*m3], m3, MPI_FLOAT, proc_id+1, 2, MPI_COMM_WORLD);
    if (proc_id != 0)
      MPI_Recv(&tgt[(offset[proc_id]-1)*m3], m3, MPI_FLOAT, proc_id-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (proc_id != 0)
      MPI_Send(&tgt[offset[proc_id]*m3], m3, MPI_FLOAT, proc_id-1, 3, MPI_COMM_WORLD);
    if (proc_id != n_proc - 1)
      MPI_Recv(&tgt[offset[proc_id+1]*m3], m3, MPI_FLOAT, proc_id+1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // --- Adaptive convergence check ---
    // Each rank computes its local partial residual; MPI_Reduce sums them on
    // rank 0; rank 0 checks threshold and broadcasts the decision to all.
    if (eps > 0.0f) {
      float local_err[3] = {0.0f, 0.0f, 0.0f};
      for (int id = offset[proc_id]*M; id < offset[proc_id+1]*M; ++id) {
        if (mask[id]) {
          int off3 = id*3;
          int id0=off3-m3, id1=off3-3, id2=off3+3, id3=off3+m3;
          local_err[0] += std::abs(grad[off3+0]+tgt[id0+0]+tgt[id1+0]+tgt[id2+0]+tgt[id3+0]-tgt[off3+0]*4.0);
          local_err[1] += std::abs(grad[off3+1]+tgt[id0+1]+tgt[id1+1]+tgt[id2+1]+tgt[id3+1]-tgt[off3+1]*4.0);
          local_err[2] += std::abs(grad[off3+2]+tgt[id0+2]+tgt[id1+2]+tgt[id2+2]+tgt[id3+2]-tgt[off3+2]*4.0);
        }
      }
      float global_err[3] = {0.0f, 0.0f, 0.0f};
      MPI_Reduce(local_err, global_err, 3, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

      if (proc_id == 0 && global_mask_count > 0) {
        float mean = (global_err[0]+global_err[1]+global_err[2]) / (3.0f * global_mask_count);
        converged = (mean < eps) ? 1 : 0;
      }
      MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  calc_error();
  if (proc_id == 0) {
    for (int i = 0; i < N * m3; ++i)
      imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
    return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
  } else {
    return std::make_tuple(py::array({1, 1, 3}, imgbuf), py::array(3, err));
  }
}
