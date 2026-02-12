

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS CliffDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(CliffDialect CliffDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS CliffTypes.td)
mlir_tablegen(CliffTypes.h.inc -gen-typedef-decls -dialect=diss)
mlir_tablegen(CliffTypes.cpp.inc -gen-typedef-defs -dialect=diss)


add_public_tablegen_target(CliffordIRTableGen)