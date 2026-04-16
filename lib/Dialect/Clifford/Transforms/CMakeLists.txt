add_clifford_library(CliffordPasses
    GeometricTypeConversion.cpp
    RewriteExponential.cpp
    
    DEPENDS
    CliffPassesPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
)