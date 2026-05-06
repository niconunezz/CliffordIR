#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"
#include "clifford/Conversion/CliffGPUToLLVM/Utility.h"
#include "clifford/Conversion/CliffGPUToLLVM/PopulatePatterns.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::clg;

namespace {


class LoadOpPattern : public OpConversionPattern<LoadOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(LoadOp op, LoadOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        auto pointer = op.getPtr();
        auto typeConverter = getTypeConverter();
        auto outTensor = dyn_cast<RankedTensorType>(op.getValue().getType());
        // delete
        if (!outTensor) {
            llvm::errs() << "getValue() doesnt return what it should!!!\n";
            return failure();
        }
        auto outTensorShape = outTensor.getShape();

        auto rank = outTensorShape.size();
        SmallVector<unsigned> order(rank);
        std::iota(order.begin(), order.end(), 0);
        std::reverse(order.begin(), order.end());

        auto llAttr = dyn_cast<LinearEncodingAttr>(outTensor.getEncoding());
        auto ll = llAttr.getLinearLayout();
        Value threadIdx = NVVM::ThreadIdXOp::create(rewriter, loc, rewriter.getI32Type());
        Value blockIdx = NVVM::BlockIdXOp::create(rewriter, loc, rewriter.getI32Type());
        Value blockDim = NVVM::BlockDimXOp::create(rewriter, loc, rewriter.getI32Type());
        
        Value c31 = b.i32_val(31);
        Value c32 = b.i32_val(5);

        Value laneId = b.and_(threadIdx, c31);
        Value warpId = b.ashr_(threadIdx, c32);

        auto kReg = StringAttr::get(ctx, "register");
        auto kLane = StringAttr::get(ctx, "lane");
        auto kWarp = StringAttr::get(ctx, "warp");

        // i, j
        auto logicalIndices = applyLinearLayout(loc, ll, {{kReg, {}}, {kLane, laneId}, {kWarp, warpId}},
                                     typeConverter, rewriter);

        SmallVector<Value> logicalIndicesVals;
        for (auto &[_, value] : logicalIndices) {
            logicalIndicesVals.push_back(value);
        }

        Value ptr_offset = b.i32_val(0);
        Value stride = b.i32_val(1);
        uint32_t idx = 0;

        ValueRange values(logicalIndicesVals);
        Value address = b.gep(LLVM::LLVMPointerType::get(ctx), rewriter.getF32Type(), pointer, values);

        SmallVector<Type> newResultTypes;
        if (failed(typeConverter->convertTypes(op.getValue(), newResultTypes))) 
            return rewriter.notifyMatchFailure(op, "failed to convert return type");
        auto resTy = newResultTypes[0];

        auto res = b.load(rewriter.getF32Type(), address);
        return success();

    }
};


};

void mlir::cliff::populateLoadStoreOpPatterns(CliffGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
    MLIRContext *context = patterns.getContext();
    patterns.insert<LoadOpPattern>(typeConverter, context);
}