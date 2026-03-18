#include "mlir/Transforms/DialectConversion.h"


#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include "clifford/Conversion/CliffToCliffGPU/Passes.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/Transforms/CliffGPUConversion.h"


using namespace mlir;
using namespace mlir::clg;


CliffGPUTypeConverter::CliffGPUTypeConverter(MLIRContext *ctx, int numWarp,
     int threadsPerWarp) : context(ctx), numWarps(numWarps), threadsPerWarp(threadsPerWarp) {
    
    
    addConversion([](Type type) { return type; });
    addConversion([this](RankedTensorType tensorType) {
        if (tensorType.getEncoding()) {
            return tensorType;
        }

        auto shape = tensorType.getShape();
        LinearEncodingAttr encoding = getDefaultGlobalEncoding(this->context, shape);

        tensorType.cloneWithEncoding(encoding);

        return tensorType;
    });
}