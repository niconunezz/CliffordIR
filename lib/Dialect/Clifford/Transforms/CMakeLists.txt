add_clifford_library(CliffordPasses
    Passes.cpp

    DEPENDS
    CliffPassesPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
)