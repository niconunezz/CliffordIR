set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS CliffGPUOps.td)
mlir_tablegen(CliffGPUOps.h.inc -gen-op-decls)
mlir_tablegen(CliffGPUOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS CliffGPUDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(CliffGPUDialect CliffGPUDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS CliffGPUTypes.td)
mlir_tablegen(CliffGPUTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CliffGPUTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS CliffGPUAttrDefs.td)
mlir_tablegen(CliffGPUAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(CliffGPUAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(CliffGPUIRTableGen)