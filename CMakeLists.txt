cmake_minimum_required(VERSION 3.20.0)

project(CliffordIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/mlir")

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})


# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})


list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Utilities
function(add_clifford_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )

  # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_clifford_object)

set_property(GLOBAL PROPERTY CLIFFORD_LIBS "")
function(add_clifford_library name)
  set_property(GLOBAL APPEND PROPERTY CLIFFORD_LIBS ${name})
  add_clifford_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

set_property(GLOBAL PROPERTY CLIFFORD_PLUGINS "")
function(add_clifford_plugin name)
  set_property(GLOBAL APPEND PROPERTY CLIFFORD_PLUGINS ${name})
  add_clifford_object(${name} ${ARGN})
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
