"""Processor abstractions and backend selection for PIE solvers."""

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from fpie import np_solver

DEFAULT_CPU_CAP = 8


def _default_cpu_count() -> int:
    candidates: list[int] = []
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        candidates.append(cpu_count)
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if not candidates:
        return 1
    return max(1, min(min(candidates), DEFAULT_CPU_CAP))


CPU_COUNT = _default_cpu_count()
DEFAULT_BACKEND = "numpy"
ALL_BACKEND = ["numpy"]
MPI: Any | None = None

try:
    from fpie import numba_solver
    ALL_BACKEND += ["numba"]
    DEFAULT_BACKEND = "numba"
except ImportError:
    numba_solver = None  # type: ignore

try:
    from fpie import taichi_solver
    ALL_BACKEND += ["taichi-cpu", "taichi-gpu"]
    DEFAULT_BACKEND = "taichi-cpu"
except ImportError:
    taichi_solver = None  # type: ignore

try:
    from fpie import core_gcc  # type: ignore
    DEFAULT_BACKEND = "gcc"
    ALL_BACKEND.append("gcc")
except ImportError:
    core_gcc = None

try:
    from fpie import core_openmp  # type: ignore
    DEFAULT_BACKEND = "openmp"
    ALL_BACKEND.append("openmp")
except ImportError:
    core_openmp = None

try:
    from mpi4py import MPI as _MPI
    from fpie import core_mpi  # type: ignore
    MPI = _MPI
    ALL_BACKEND.append("mpi")
except ImportError:
    MPI = None
    core_mpi = None

try:
    from fpie import core_cuda  # type: ignore
    DEFAULT_BACKEND = "cuda"
    ALL_BACKEND.append("cuda")
except ImportError:
    core_cuda = None

# --- Hybrid: MPI + OpenMP (requires both to be available) ---
try:
    from mpi4py import MPI as _MPI2
    from fpie import core_hybrid  # type: ignore
    if MPI is None:
        MPI = _MPI2
    ALL_BACKEND.append("hybrid")
except ImportError:
    core_hybrid = None


class BaseProcessor(ABC):
    def __init__(self, gradient: str, rank: int, backend: str, core: Any | None):
        if core is None:
            error_msg = {
                "numpy":   "Please run `pip install numpy`.",
                "numba":   "Please run `pip install numba`.",
                "gcc":     "Please install cmake and gcc.",
                "openmp":  "Please make sure your gcc supports `-fopenmp`.",
                "mpi":     "Please install MPI and run `pip install mpi4py`.",
                "cuda":    "Please make sure nvcc and cuda libraries are available.",
                "taichi":  "Please run `pip install taichi`.",
                "hybrid":  "Please install MPI + OpenMP and rebuild with cmake.",
            }
            print(error_msg.get(backend.split("-")[0], f"Backend {backend} unavailable."))
            raise AssertionError(f"Invalid backend {backend}.")

        self.gradient = gradient
        self.rank = rank
        self.backend = backend
        self.core = core
        self.root = rank == 0

    def mixgrad(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.gradient == "src":
            return a
        if self.gradient == "avg":
            return (a + b) / 2
        mask = np.abs(a) < np.abs(b)
        a[mask] = b[mask]
        return a

    @abstractmethod
    def reset(self, src, mask, tgt, mask_on_src, mask_on_tgt) -> int:
        pass

    def sync(self) -> None:
        self.core.sync()

    @abstractmethod
    def step(self, iteration: int, eps: float = 0.0):
        pass


class EquProcessor(BaseProcessor):
    def __init__(
        self,
        gradient: str = "max",
        backend: str = DEFAULT_BACKEND,
        n_cpu: int = CPU_COUNT,
        min_interval: int = 100,
        block_size: int = 1024,
    ):
        core: Any | None = None
        rank = 0

        if backend == "numpy":
            core = np_solver.EquSolver()
        elif backend == "numba" and numba_solver is not None:
            core = numba_solver.EquSolver()
        elif backend == "gcc":
            core = core_gcc.EquSolver()
        elif backend == "openmp" and core_openmp is not None:
            core = core_openmp.EquSolver(n_cpu)
        elif backend == "mpi" and core_mpi is not None:
            assert MPI is not None
            core = core_mpi.EquSolver(min_interval)
            rank = MPI.COMM_WORLD.Get_rank()
        elif backend == "cuda" and core_cuda is not None:
            core = core_cuda.EquSolver(block_size)
        elif backend == "hybrid" and core_hybrid is not None:
            assert MPI is not None
            core = core_hybrid.EquSolver(n_cpu, min_interval)
            rank = MPI.COMM_WORLD.Get_rank()
        elif backend.startswith("taichi") and taichi_solver is not None:
            core = taichi_solver.EquSolver(backend, n_cpu, block_size)

        super().__init__(gradient, rank, backend, core)

    def mask2index(self, mask):
        x, y = np.nonzero(mask)
        max_id = x.shape[0] + 1
        index = np.zeros((max_id, 3))
        ids = self.core.partition(mask)
        ids[mask == 0] = 0
        index = ids[x, y].argsort()
        return ids, max_id, x[index], y[index]

    def reset(self, src, mask, tgt, mask_on_src, mask_on_tgt) -> int:
        assert self.root
        if len(mask.shape) == 3:
            mask = mask.mean(-1)
        mask = (mask >= 128).astype(np.int32)
        mask[0] = 0; mask[-1] = 0; mask[:, 0] = 0; mask[:, -1] = 0

        x, y = np.nonzero(mask)
        x0, x1 = x.min() - 1, x.max() + 2
        y0, y1 = y.min() - 1, y.max() + 2
        mask_on_src = (x0 + mask_on_src[0], y0 + mask_on_src[1])
        mask_on_tgt = (x0 + mask_on_tgt[0], y0 + mask_on_tgt[1])
        mask = mask[x0:x1, y0:y1]
        ids, max_id, index_x, index_y = self.mask2index(mask)

        src_x, src_y = index_x + mask_on_src[0], index_y + mask_on_src[1]
        tgt_x, tgt_y = index_x + mask_on_tgt[0], index_y + mask_on_tgt[1]

        src_C = src[src_x, src_y].astype(np.float32)
        src_U = src[src_x-1, src_y].astype(np.float32)
        src_D = src[src_x+1, src_y].astype(np.float32)
        src_L = src[src_x, src_y-1].astype(np.float32)
        src_R = src[src_x, src_y+1].astype(np.float32)
        tgt_C = tgt[tgt_x, tgt_y].astype(np.float32)
        tgt_U = tgt[tgt_x-1, tgt_y].astype(np.float32)
        tgt_D = tgt[tgt_x+1, tgt_y].astype(np.float32)
        tgt_L = tgt[tgt_x, tgt_y-1].astype(np.float32)
        tgt_R = tgt[tgt_x, tgt_y+1].astype(np.float32)

        grad = (
            self.mixgrad(src_C-src_L, tgt_C-tgt_L)
            + self.mixgrad(src_C-src_R, tgt_C-tgt_R)
            + self.mixgrad(src_C-src_U, tgt_C-tgt_U)
            + self.mixgrad(src_C-src_D, tgt_C-tgt_D)
        )

        A = np.zeros((max_id, 4), np.int32)
        X = np.zeros((max_id, 3), np.float32)
        B = np.zeros((max_id, 3), np.float32)

        X[1:] = tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1]]
        A[1:, 0] = ids[index_x-1, index_y]
        A[1:, 1] = ids[index_x+1, index_y]
        A[1:, 2] = ids[index_x, index_y-1]
        A[1:, 3] = ids[index_x, index_y+1]
        B[1:] = grad
        m = (mask[index_x-1, index_y]==0).astype(float).reshape(-1,1)
        B[1:] += m * tgt[index_x+mask_on_tgt[0]-1, index_y+mask_on_tgt[1]]
        m = (mask[index_x, index_y-1]==0).astype(float).reshape(-1,1)
        B[1:] += m * tgt[index_x+mask_on_tgt[0], index_y+mask_on_tgt[1]-1]
        m = (mask[index_x, index_y+1]==0).astype(float).reshape(-1,1)
        B[1:] += m * tgt[index_x+mask_on_tgt[0], index_y+mask_on_tgt[1]+1]
        m = (mask[index_x+1, index_y]==0).astype(float).reshape(-1,1)
        B[1:] += m * tgt[index_x+mask_on_tgt[0]+1, index_y+mask_on_tgt[1]]

        self.tgt = tgt.copy()
        self.tgt_index = (index_x+mask_on_tgt[0], index_y+mask_on_tgt[1])
        self.core.reset(max_id, A, X, B)
        return max_id

    def step(self, iteration: int, eps: float = 0.0):
        EPS_BACKENDS = {"openmp", "mpi", "hybrid"}
        if self.backend in EPS_BACKENDS:
            result = self.core.step(iteration, eps)
        else:
            result = self.core.step(iteration)
        if self.root:
            x, err = result
            self.tgt[self.tgt_index] = x[1:]
            return self.tgt, err
        return None


class GridProcessor(BaseProcessor):
    def __init__(
        self,
        gradient: str = "max",
        backend: str = DEFAULT_BACKEND,
        n_cpu: int = CPU_COUNT,
        min_interval: int = 100,
        block_size: int = 1024,
        grid_x: int = 8,
        grid_y: int = 8,
    ):
        core: Any | None = None
        rank = 0

        if backend == "numpy":
            core = np_solver.GridSolver()
        elif backend == "numba" and numba_solver is not None:
            core = numba_solver.GridSolver()
        elif backend == "gcc":
            core = core_gcc.GridSolver(grid_x, grid_y)
        elif backend == "openmp" and core_openmp is not None:
            core = core_openmp.GridSolver(grid_x, grid_y, n_cpu)
        elif backend == "mpi" and core_mpi is not None:
            assert MPI is not None
            core = core_mpi.GridSolver(min_interval)
            rank = MPI.COMM_WORLD.Get_rank()
        elif backend == "cuda" and core_cuda is not None:
            core = core_cuda.GridSolver(grid_x, grid_y)
        elif backend == "hybrid" and core_hybrid is not None:
            assert MPI is not None
            core = core_hybrid.GridSolver(grid_x, grid_y, n_cpu, min_interval)
            rank = MPI.COMM_WORLD.Get_rank()
        elif backend.startswith("taichi") and taichi_solver is not None:
            core = taichi_solver.GridSolver(grid_x, grid_y, backend, n_cpu, block_size)

        super().__init__(gradient, rank, backend, core)

    def reset(self, src, mask, tgt, mask_on_src, mask_on_tgt) -> int:
        assert self.root
        if len(mask.shape) == 3:
            mask = mask.mean(-1)
        mask = (mask >= 128).astype(np.int32)
        mask[0] = 0; mask[-1] = 0; mask[:, 0] = 0; mask[:, -1] = 0

        x, y = np.nonzero(mask)
        x0, x1 = x.min()-1, x.max()+2
        y0, y1 = y.min()-1, y.max()+2
        mask = mask[x0:x1, y0:y1]
        max_id = int(np.prod(mask.shape))

        src_crop = src[mask_on_src[0]+x0:mask_on_src[0]+x1,
                       mask_on_src[1]+y0:mask_on_src[1]+y1].astype(np.float32)
        tgt_crop = tgt[mask_on_tgt[0]+x0:mask_on_tgt[0]+x1,
                       mask_on_tgt[1]+y0:mask_on_tgt[1]+y1].astype(np.float32)
        grad = np.zeros([*mask.shape, 3], np.float32)
        grad[1:]   += self.mixgrad(src_crop[1:]-src_crop[:-1], tgt_crop[1:]-tgt_crop[:-1])
        grad[:-1]  += self.mixgrad(src_crop[:-1]-src_crop[1:], tgt_crop[:-1]-tgt_crop[1:])
        grad[:,1:]  += self.mixgrad(src_crop[:,1:]-src_crop[:,:-1], tgt_crop[:,1:]-tgt_crop[:,:-1])
        grad[:,:-1] += self.mixgrad(src_crop[:,:-1]-src_crop[:,1:], tgt_crop[:,:-1]-tgt_crop[:,1:])
        grad[mask == 0] = 0

        self.x0, self.x1 = mask_on_tgt[0]+x0, mask_on_tgt[0]+x1
        self.y0, self.y1 = mask_on_tgt[1]+y0, mask_on_tgt[1]+y1
        self.tgt = tgt.copy()
        self.core.reset(max_id, mask, tgt_crop, grad)
        return max_id

    def step(self, iteration: int, eps: float = 0.0):
        EPS_BACKENDS = {"openmp", "mpi", "hybrid"}
        if self.backend in EPS_BACKENDS:
            result = self.core.step(iteration, eps)
        else:
            result = self.core.step(iteration)
        if self.root:
            tgt, err = result
            self.tgt[self.x0:self.x1, self.y0:self.y1] = tgt
            return self.tgt, err
        return None
