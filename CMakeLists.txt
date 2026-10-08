cmake_minimum_required(VERSION 3.20.0)

project(CliffordIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)
set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)


add_link_options(-fuse-ld=mold)
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/mlir")

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
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

include_directories(${PYTHON_SRC_PATH})
add_subdirectory(include)
add_subdirectory(lib)

# python setup
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/clifford/_C" CACHE PATH "" FORCE)
set(DEV_MODULE Development.Module)
find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

  
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_build_library(nanobind-static)

  if(NOT TARGET Python::Module)
  message(WARNING "PYTHON NOT FOUND")
  # add_library(Python::Module ALIAS Python3::Module)
  # set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
  # set(Python_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
  # set(Python_SOABI "${Python3_SOABI}")
  # set(Python_VERSION "${Python3_VERSION}")
  # set(Python_INTERPRETER_ID "${Python3_INTERPRETER_ID}")
  endif()
  
  
  target_link_libraries(nanobind-static PRIVATE Python::Module)
  add_library(clifford SHARED ${PYTHON_SRC_PATH}/main.cc
  ${PYTHON_SRC_PATH}/ir.cc
  ${PYTHON_SRC_PATH}/passes.cc
  ${PYTHON_SRC_PATH}/llvm.cc)
  # add_dependencies(clifford CliffordIRTableGen CliffGPUIRTableGen)

  get_property(clifford_libs GLOBAL PROPERTY CLIFFORD_LIBS)
  get_property(clifford_plugins GLOBAL PROPERTY CLIFFORD_PLUGINS)
  set(CLIFFORD_LIBRARIES
    ${clifford_libs}
    ${clifford_plugins}
    # mlir
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRIR
    MLIRControlFlowToLLVM
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRMathToLLVM
    MLIRROCDLToLLVMIRTranslation
    MLIRGPUDialect
    MLIRSCFToControlFlow
    MLIRIndexToLLVM
    MLIRGPUToROCDLTransforms
    MLIRUBToLLVM
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMToLLVMIRTranslation
    # LLVM
    LLVMBitWriter
    LLVMPasses
    LLVMNVPTXCodeGen
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser

    Python::Module
    nanobind-static
  )

  target_link_libraries(clifford PRIVATE ${CLIFFORD_LIBRARIES})

add_subdirectory(bin)
add_subdirectory(unittest)

