add_clifford_library(CliffGPUIR

    Dialect.cpp

    DEPENDS
    CliffGPUIRTableGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
)