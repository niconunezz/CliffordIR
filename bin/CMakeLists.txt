get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(cliford_libs GLOBAL PROPERTY CLIFORD_LIBS)

add_llvm_executable(cliff-opt cliff-opt.cpp)
target_link_libraries(cliff-opt PRIVATE
${cliford_libs}
CliffordIR
${dialect_libs}
${conversion_libs}
MLIROptLib
MLIRPass
MLIRRegisterAllDialects
MLIRRegisterAllPasses
MLIRTransforms
)

mlir_check_all_link_libraries(cliff-opt)
