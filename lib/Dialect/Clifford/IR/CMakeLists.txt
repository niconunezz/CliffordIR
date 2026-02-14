add_clifford_library(CliffordIR
    Dialect.cpp
    CliffOps.cpp
    CliffTypes.cpp

    DEPENDS
    CliffordIRTableGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
)