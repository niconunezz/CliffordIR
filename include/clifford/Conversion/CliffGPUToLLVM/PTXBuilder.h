#ifndef CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_PTXBUILDER_H
#define CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_PTXBUILDER_H 

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::clg {

    struct PTXBuilder {
        PTXBuilder(Location loc, OpBuilder &builder) : loc(loc), builder(builder) {};

        Location loc;
        OpBuilder builder; 
    }

} //end namespace mlir::clg 

#endif // CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_PTXBUILDER_H