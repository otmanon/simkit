# Standalone build of the simkit C++ core.
#
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build cpp/build
#   ctest --test-dir cpp/build
#
# Produces the static library ``simkit`` (link it and add ``include/`` to use
# the C++ API directly), the Python module ``simkit._cpp`` when pybind11 is
# found (SIMKIT_BUILD_PYTHON) and a small test executable (SIMKIT_BUILD_TESTS).
# ``pip install .`` at the repository root builds the Python module through
# setup.py instead and does not need CMake.
cmake_minimum_required(VERSION 3.15)
project(simkit_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/bigobj /EHsc)
endif()

option(SIMKIT_BUILD_PYTHON "Build the simkit._cpp pybind11 module" ON)
option(SIMKIT_BUILD_TESTS "Build the C++ tests" ON)

# ---- Eigen -----------------------------------------------------------------
if(DEFINED ENV{EIGEN3_INCLUDE_DIR} AND EXISTS "$ENV{EIGEN3_INCLUDE_DIR}/Eigen/Core")
  set(SIMKIT_EIGEN_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen/Eigen/Core")
  set(SIMKIT_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../build/eigen/eigen-3.4.0/Eigen/Core")
  set(SIMKIT_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/eigen/eigen-3.4.0")  # setup.py's download
else()
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
  get_target_property(SIMKIT_EIGEN_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
endif()
message(STATUS "simkit: using Eigen from ${SIMKIT_EIGEN_DIR}")

# ---- library ---------------------------------------------------------------
find_package(Threads REQUIRED)
file(GLOB_RECURSE SIMKIT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(simkit STATIC ${SIMKIT_SOURCES})
target_include_directories(simkit PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${SIMKIT_EIGEN_DIR}")
target_compile_definitions(simkit PUBLIC EIGEN_MPL2_ONLY)
target_link_libraries(simkit PUBLIC Threads::Threads)
set_target_properties(simkit PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- python module ---------------------------------------------------------
if(SIMKIT_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module QUIET)
  if(Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_pybind11_dir)
      list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
    endif()
    find_package(pybind11 CONFIG QUIET)
  endif()
  if(pybind11_FOUND)
    pybind11_add_module(_cpp "${CMAKE_CURRENT_SOURCE_DIR}/bindings/simkit_cpp.cpp")
    target_link_libraries(_cpp PRIVATE simkit)
    # Drop the built module next to the Python package so ``import simkit``
    # picks it up from a source checkout (.pyd is a runtime output on Windows).
    set(_simkit_pkg_dir "${CMAKE_CURRENT_SOURCE_DIR}/../simkit")
    set_target_properties(_cpp PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${_simkit_pkg_dir}"
      RUNTIME_OUTPUT_DIRECTORY "${_simkit_pkg_dir}")
    foreach(_cfg Debug Release RelWithDebInfo MinSizeRel)
      string(TOUPPER "${_cfg}" _CFG)
      set_target_properties(_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${_simkit_pkg_dir}"
        RUNTIME_OUTPUT_DIRECTORY_${_CFG} "${_simkit_pkg_dir}")
    endforeach()
    message(STATUS "simkit: building the Python module simkit._cpp")
  else()
    message(STATUS "simkit: pybind11 not found, skipping the Python module (pip install pybind11)")
  endif()
endif()

# ---- tests -----------------------------------------------------------------
if(SIMKIT_BUILD_TESTS)
  enable_testing()
  add_executable(simkit_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_simkit.cpp")
  target_link_libraries(simkit_tests PRIVATE simkit)
  add_test(NAME simkit_tests COMMAND simkit_tests)
endif()
