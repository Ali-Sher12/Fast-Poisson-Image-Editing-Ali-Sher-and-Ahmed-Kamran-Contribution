#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "solver.h"

HybridGridSolver::HybridGridSolver(int grid_x, int grid_y, int n_cpu,
                                   int min_interval)
    : imgbuf(NULL), min_interval(min_interval), GridSolver(grid_x, grid_y) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  offset = new int[n_proc + 1];
  omp_set_num_threads(n_cpu);
}

HybridGridSolver::~HybridGridSolver() {
  if (imgbuf != NULL) delete[] imgbuf;
  delete[] offset;
}

void HybridGridSolver::post_reset() {
  if (imgbuf != NULL) delete[] imgbuf;
  imgbuf = new unsigned char[N * m3];

  std::vector<int> row_pixels(N, 0);
  int n_mask_pixels = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      if (mask[i * M + j]) { ++row_pixels[i]; ++n_mask_pixels; }

  offset[0] = 0;
  if (n_mask_pixels == 0) {
    int extra = N % n_proc;
    for (int i = 0; i < n_proc; ++i)
      offset[i+1] = offset[i] + N/n_proc + (i < extra);
    return;
  }
  int target = (n_mask_pixels + n_proc - 1) / n_proc;
  int cumsum = 0, rank = 1;
  for (int i = 0; i < N && rank < n_proc; ++i) {
    cumsum += row_pixels[i];
    if (cumsum >= target * rank) offset[rank++] = i + 1;
  }
  while (rank <= n_proc) offset[rank++] = N;
}

void HybridGridSolver::sync() {
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (proc_id > 0) {
    if (mask != NULL) { delete[] mask; delete[] tgt; delete[] grad; delete[] imgbuf; }
    m3   = M * 3;
    mask = new int[N * M];
    tgt  = new float[N * m3];
    grad = new float[N * m3];
    imgbuf = new unsigned char[N * m3];
  }
  MPI_Bcast(mask,   N * M,  MPI_INT,   0, MPI_COMM_WORLD);
  MPI_Bcast(tgt,    N * m3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(grad,   N * m3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(offset, n_proc + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

inline void HybridGridSolver::update_equation(int id) {
  int off3 = id*3, id0=off3-m3, id1=off3-3, id2=off3+3, id3=off3+m3;
  tgt[off3+0]=(grad[off3+0]+tgt[id0+0]+tgt[id1+0]+tgt[id2+0]+tgt[id3+0])/4.0;
  tgt[off3+1]=(grad[off3+1]+tgt[id0+1]+tgt[id1+1]+tgt[id2+1]+tgt[id3+1])/4.0;
  tgt[off3+2]=(grad[off3+2]+tgt[id0+2]+tgt[id1+2]+tgt[id2+2]+tgt[id3+2])/4.0;
}

void HybridGridSolver::calc_error() {
  memset(err, 0, sizeof(err));
  for (int id = offset[proc_id]*M; id < offset[proc_id+1]*M; ++id) {
    if (mask[id]) {
      int off3=id*3, id0=off3-m3, id1=off3-3, id2=off3+3, id3=off3+m3;
      err[0]+=std::abs(grad[off3+0]+tgt[id0+0]+tgt[id1+0]+tgt[id2+0]+tgt[id3+0]-tgt[off3+0]*4.0);
      err[1]+=std::abs(grad[off3+1]+tgt[id0+1]+tgt[id1+1]+tgt[id2+1]+tgt[id3+1]-tgt[off3+1]*4.0);
      err[2]+=std::abs(grad[off3+2]+tgt[id0+2]+tgt[id1+2]+tgt[id2+2]+tgt[id3+2]-tgt[off3+2]*4.0);
    }
  }
  if (proc_id == 0) {
    float tmp[3];
    for (int j = 1; j < n_proc; ++j) {
      MPI_Recv(&tgt[offset[j]*m3], (offset[j+1]-offset[j])*m3,
               MPI_FLOAT, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&tmp, 3, MPI_FLOAT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      err[0]+=tmp[0]; err[1]+=tmp[1]; err[2]+=tmp[2];
    }
  } else {
    MPI_Send(&tgt[offset[proc_id]*m3], (offset[proc_id+1]-offset[proc_id])*m3,
             MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(&err, 3, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
HybridGridSolver::step(int iteration, float eps) {
  int converged = 0;

  int local_mask = 0;
  for (int id = offset[proc_id]*M; id < offset[proc_id+1]*M; ++id)
    if (mask[id]) ++local_mask;
  int global_mask = 0;
  MPI_Allreduce(&local_mask, &global_mask, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  for (int i = 0; i < iteration && !converged; i += min_interval) {
    // Two-level Jacobi: MPI owns rows, OpenMP owns pixels within rows
    for (int s = 0; s < min_interval; ++s) {
      int row_start = offset[proc_id]   * M;
      int row_end   = offset[proc_id+1] * M;
#pragma omp parallel for schedule(static)
      for (int k = row_start; k < row_end; ++k)
        if (mask[k]) update_equation(k);
    }

    // Halo exchange
    if (proc_id != n_proc-1)
      MPI_Send(&tgt[(offset[proc_id+1]-1)*m3], m3, MPI_FLOAT, proc_id+1, 2, MPI_COMM_WORLD);
    if (proc_id != 0)
      MPI_Recv(&tgt[(offset[proc_id]-1)*m3], m3, MPI_FLOAT, proc_id-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (proc_id != 0)
      MPI_Send(&tgt[offset[proc_id]*m3], m3, MPI_FLOAT, proc_id-1, 3, MPI_COMM_WORLD);
    if (proc_id != n_proc-1)
      MPI_Recv(&tgt[offset[proc_id+1]*m3], m3, MPI_FLOAT, proc_id+1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Adaptive convergence — use separate scalars to avoid array reduction issue
    if (eps > 0.0f) {
      float le0 = 0.0f, le1 = 0.0f, le2 = 0.0f;
      int row_start = offset[proc_id]   * M;
      int row_end   = offset[proc_id+1] * M;
#pragma omp parallel for schedule(static) reduction(+:le0,le1,le2)
      for (int id = row_start; id < row_end; ++id) {
        if (mask[id]) {
          int off3=id*3, id0=off3-m3, id1=off3-3, id2=off3+3, id3=off3+m3;
          le0 += std::abs(grad[off3+0]+tgt[id0+0]+tgt[id1+0]+tgt[id2+0]+tgt[id3+0]-tgt[off3+0]*4.0);
          le1 += std::abs(grad[off3+1]+tgt[id0+1]+tgt[id1+1]+tgt[id2+1]+tgt[id3+1]-tgt[off3+1]*4.0);
          le2 += std::abs(grad[off3+2]+tgt[id0+2]+tgt[id1+2]+tgt[id2+2]+tgt[id3+2]-tgt[off3+2]*4.0);
        }
      }
      float local_err[3]  = {le0, le1, le2};
      float global_err[3] = {0.f, 0.f, 0.f};
      MPI_Reduce(local_err, global_err, 3, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

      if (proc_id == 0 && global_mask > 0) {
        float mean = (global_err[0]+global_err[1]+global_err[2]) / (3.0f * global_mask);
        converged = (mean < eps) ? 1 : 0;
      }
      MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  calc_error();
  if (proc_id == 0) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N * m3; ++i)
      imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
    return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
  } else {
    return std::make_tuple(py::array({1, 1, 3}, imgbuf), py::array(3, err));
  }
}
