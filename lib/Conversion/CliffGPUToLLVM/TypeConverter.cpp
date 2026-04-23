#include "mlir/Transforms/DialectConversion.h"


#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"
#include "clifford/Conversion/CliffGPUToLLVM/TypeConverter.h"


using namespace mlir;
using namespace mlir::clg;

CliffGPUToLLVMTypeConverter::CliffGPUToLLVMTypeConverter(MLIRContext *ctx) : context(ctx) {

    addConversion([ctx](cliff::Cliff_MultivectorType mvector) {
        int numComponents = __builtin_popcountll(mvector.getMask());
        SmallVector<Type, 4> types(numComponents, mvector.getDtype());
        return LLVM::LLVMStructType::getLiteral(ctx, types);
    });

    addConversion([&](RankedTensorType type) {
        auto ctx = type.getContext();
        Type eltType = convertType(type.getElementType());
        LinearEncodingAttr llAttr = dyn_cast<LinearEncodingAttr>(type.getEncoding());
        assert(llAttr && "Encodings should always be linear encodings!");

        //! this doesnt make sense rn, must think about
        // unsigned elPerThread = llAttr.getTotalElemsPerThread(type.getShape());
        SmallVector<Type, 4> types(1, eltType);
        return LLVM::LLVMStructType::getLiteral(ctx, types);
    });
}

CliffGPUToLLVMConversionTarget::CliffGPUToLLVMConversionTarget(MLIRContext &ctx,
     const TypeConverter &typeConverter) : ConversionTarget(ctx) {

    addLegalOp<ModuleOp>();
    addIllegalOp<cliff::Rotate, cliff::Translate>();
    addLegalDialect<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect, mlir::math::MathDialect>();
    addIllegalDialect<cliff::CliffDialect, clg::CliffGPUDialect>();

}
