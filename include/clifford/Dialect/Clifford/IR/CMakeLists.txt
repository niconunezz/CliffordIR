

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS CliffOps.td)
mlir_tablegen(CliffOps.h.inc -gen-op-decls)
mlir_tablegen(CliffOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS CliffDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(CliffDialect CliffDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS CliffTypes.td)
mlir_tablegen(CliffTypes.h.inc -gen-typedef-decls -dialect=diss)
mlir_tablegen(CliffTypes.cpp.inc -gen-typedef-defs -dialect=diss)

set(LLVM_TARGET_DEFINITIONS CliffAttrDefs.td)
mlir_tablegen(CliffAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(CliffAttrDefs.cpp.inc -gen-attrdef-defs)


add_public_tablegen_target(CliffordIRTableGen)