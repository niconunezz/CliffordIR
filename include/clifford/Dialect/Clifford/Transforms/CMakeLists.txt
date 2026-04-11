set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(CliffPasses.h.inc -gen-pass-decls -name Passes)

add_public_tablegen_target(CliffPassesPassIncGen)
