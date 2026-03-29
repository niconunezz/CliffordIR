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


CliffGPUTypeConverter::CliffGPUTypeConverter(MLIRContext *ctx, int numWarps,
     int threadsPerWarp) : context(ctx), numWarps(numWarps), threadsPerWarp(threadsPerWarp) {
    
    
    addConversion([](Type type) { return type; });
    addConversion([this](RankedTensorType tensorType) {
        if (tensorType.getEncoding()) {
            return tensorType;
        }

        auto shape = tensorType.getShape();
        LinearEncodingAttr encoding = getDefaultGlobalEncoding(this->context, this->numWarps, this->threadsPerWarp ,shape);

        tensorType.cloneWithEncoding(encoding);

        return tensorType;
    });
};


CliffGPUConversionTarget::CliffGPUConversionTarget(MLIRContext &ctx, const TypeConverter &typeConverter) : ConversionTarget(ctx) {

    addLegalDialect<clg::CliffGPUDialect>();
    addDynamicallyLegalDialect<cliff::CliffDialect>
        ([&](Operation *op) {return isDynamicallyLegal(op, typeConverter); });

    addDynamicallyLegalOp<cliff::FuncOp>([&](cliff::FuncOp op) -> bool {
        for (auto arg : op.getArguments()) {
            if (auto tensor = dyn_cast<RankedTensorType>(arg.getType())) {
                if (!tensor.getEncoding()) 
                    return false;
            }
        }
        return true;
    });

};


bool CliffGPUConversionTarget::isDynamicallyLegal(
    Operation *op, const TypeConverter &typeConverter) {

    bool hasLegalRegions = true;
    for (auto &region : op->getRegions()) {
        hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
    }

    if (hasLegalRegions && typeConverter.isLegal(op)) {
        return true;
    }

    return false;

}