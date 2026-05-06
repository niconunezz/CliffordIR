add_clifford_library(CliffGPUToLLVMPass

    Utility.cpp
    CliffGPUToLLVMPass.cpp
    TypeConverter.cpp
    LoadStoreOpToLLVM.cpp
    GeometricToLLVM.cpp
    GeometricRewrites.cpp
    
    DEPENDS
    CliffordGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
    CliffGPUIR
)