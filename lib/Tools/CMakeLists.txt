add_clifford_library(CliffTools
  LinearLayouts.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect

)