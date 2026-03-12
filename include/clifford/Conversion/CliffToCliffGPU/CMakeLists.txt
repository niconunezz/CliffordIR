set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name CliffToCliffGPU)
add_public_tablegen_target(CliffordConversionPassIncGen)
