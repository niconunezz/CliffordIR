add_clifford_library(CliffToCliffGPUPass

    CliffToCliffGPUPass.cpp
    CliffToCliffGPU.cpp
    
    DEPENDS
    CliffordConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
)