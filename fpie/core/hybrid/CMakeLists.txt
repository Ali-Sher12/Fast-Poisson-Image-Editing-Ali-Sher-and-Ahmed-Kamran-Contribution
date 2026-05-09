# CMakeLists.txt for the hybrid (MPI + OpenMP) backend.
# Mirrors the structure of fpie/core/mpi/CMakeLists.txt but adds -fopenmp.

if(OpenMP_CXX_FOUND AND MPI_CXX_FOUND)
  pybind11_add_module(
    core_hybrid
    equ.cc
    grid.cc
    solver.cc
  )

  target_include_directories(core_hybrid PRIVATE
    ${CMAKE_SOURCE_DIR}/fpie/core
    ${MPI_CXX_INCLUDE_DIRS}
  )

  target_compile_options(core_hybrid PRIVATE
    ${OpenMP_CXX_FLAGS}
  )

  target_link_libraries(core_hybrid PRIVATE
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
  )

  install(TARGETS core_hybrid DESTINATION fpie)

  message(STATUS "Hybrid (MPI + OpenMP) backend enabled")
else()
  message(STATUS "Hybrid backend skipped: requires both OpenMP and MPI")
endif()
