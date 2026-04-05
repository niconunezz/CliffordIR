add_clifford_library(CliffGPUToLLVMPass

    CliffGPUToLLVMPass.cpp
    TypeConverter.cpp
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