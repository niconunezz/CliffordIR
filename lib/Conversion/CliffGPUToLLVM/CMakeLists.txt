add_clifford_library(CliffGPUToLLVMPass

    CliffGPUToLLVMPass.cpp
    # CliffGPUConversion.cpp
    
    DEPENDS
    CliffordGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
    CliffGPUIR
)