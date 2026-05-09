#include <omp.h>

#include <tuple>

#include "solver.h"

static const int EPS_CHECK_INTERVAL = 100;

OpenMPGridSolver::OpenMPGridSolver(int grid_x, int grid_y, int n_cpu)
    : imgbuf(NULL), tmp(NULL), n_mask_pixels(0), GridSolver(grid_x, grid_y) {
  omp_set_num_threads(n_cpu);
}

OpenMPGridSolver::~OpenMPGridSolver() {
  if (tmp != NULL) {
    delete[] tmp;
    delete[] imgbuf;
  }
}

void OpenMPGridSolver::post_reset() {
  if (tmp != NULL) {
    delete[] tmp;
    delete[] imgbuf;
  }
  tmp    = new float[N * m3];
  imgbuf = new unsigned char[N * m3];
  memset(tmp, 0, sizeof(float) * N * m3);

  // Count mask pixels once for normalisation in has_converged().
  n_mask_pixels = 0;
  for (int id = 0; id < N * M; ++id)
    if (mask[id]) ++n_mask_pixels;
}

inline void OpenMPGridSolver::update_equation(int id) {
  int off3 = id * 3;
  int id0 = off3 - m3, id1 = off3 - 3, id2 = off3 + 3, id3 = off3 + m3;
  tmp[off3+0] = (grad[off3+0]+tgt[id0+0]+tgt[id1+0]+tgt[id2+0]+tgt[id3+0]) / 4.0;
  tmp[off3+1] = (grad[off3+1]+tgt[id0+1]+tgt[id1+1]+tgt[id2+1]+tgt[id3+1]) / 4.0;
  tmp[off3+2] = (grad[off3+2]+tgt[id0+2]+tgt[id1+2]+tgt[id2+2]+tgt[id3+2]) / 4.0;
}

void OpenMPGridSolver::calc_error() {
#pragma omp parallel for schedule(static)
  for (int id = 0; id < N * M; ++id) {
    if (mask[id]) {
      int off3 = id*3, id0=off3-m3, id1=off3-3, id2=off3+3, id3=off3+m3;
      tmp[off3+0] = std::abs(grad[off3+0]+tgt[id0+0]+tgt[id1+0]+tgt[id2+0]+tgt[id3+0]-tgt[off3+0]*4.0);
      tmp[off3+1] = std::abs(grad[off3+1]+tgt[id0+1]+tgt[id1+1]+tgt[id2+1]+tgt[id3+1]-tgt[off3+1]*4.0);
      tmp[off3+2] = std::abs(grad[off3+2]+tgt[id0+2]+tgt[id1+2]+tgt[id2+2]+tgt[id3+2]-tgt[off3+2]*4.0);
    }
  }
  memset(err, 0, sizeof(err));
  for (int id = 0; id < N * M; ++id) {
    err[0] += tmp[id*3+0];
    err[1] += tmp[id*3+1];
    err[2] += tmp[id*3+2];
  }
}

bool OpenMPGridSolver::has_converged(float eps) {
  if (eps <= 0.0f || n_mask_pixels == 0) return false;
  calc_error();
  float mean = (err[0]+err[1]+err[2]) / (3.0f * n_mask_pixels);
  return mean < eps;
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
OpenMPGridSolver::step(int iteration, float eps) {
  int block_x = (N + grid_x - 1) / grid_x;
  int block_y = (M + grid_y - 1) / grid_y;

  for (int i = 0; i < iteration; ++i) {
#pragma omp parallel for schedule(static)
    for (int block_id = 0; block_id < block_x * block_y; ++block_id) {
      int id_x = block_id / block_y * grid_x;
      int id_y = block_id % block_y * grid_y;
      for (int j = 0, x = id_x; j < grid_x && x < N; ++j, ++x)
        for (int k = 0, y = id_y, id = x*M+id_y; k < grid_y && y < M; ++k, ++y, ++id)
          if (mask[id]) update_equation(id);
    }
#pragma omp parallel for schedule(static)
    for (int id = 0; id < N * M; ++id) {
      if (mask[id]) {
        tgt[id*3+0] = tmp[id*3+0];
        tgt[id*3+1] = tmp[id*3+1];
        tgt[id*3+2] = tmp[id*3+2];
      }
    }

    if (eps > 0.0f && (i + 1) % EPS_CHECK_INTERVAL == 0)
      if (has_converged(eps)) break;
  }

  calc_error();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; ++i)
    imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
