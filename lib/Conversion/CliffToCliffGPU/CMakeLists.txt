add_clifford_library(CliffToCliffGPUPass

    Passes.cpp

    DEPENDS
    CliffordConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
)