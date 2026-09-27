get_property(clifford_libs GLOBAL PROPERTY CLIFFORD_LIBS)

add_llvm_executable(cliff-opt cliff-opt.cpp)

target_link_libraries(cliff-opt PRIVATE
    ${clifford_libs}

    # MLIR core
    MLIROptLib
    MLIRPass
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRTransforms
)

mlir_check_all_link_libraries(cliff-opt)
