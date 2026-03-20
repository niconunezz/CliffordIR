add_clifford_library(CliffToCliffGPUPass

    CliffToCliffGPUPass.cpp
    CliffGPUConversion.cpp
    
    DEPENDS
    CliffordConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
    CliffGPUIR
)