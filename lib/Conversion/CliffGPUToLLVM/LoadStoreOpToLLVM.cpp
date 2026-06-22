#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"
#include "clifford/Conversion/CliffGPUToLLVM/Utility.h"
#include "clifford/Conversion/CliffGPUToLLVM/PopulatePatterns.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::clg;

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "load-store-to-llvm"

namespace {


class LoadOpPattern : public OpConversionPattern<LoadOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(LoadOp op,
                                  LoadOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();

        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        auto llPointer = adaptor.getPtr();

        auto typeConverter = getTypeConverter();
        RankedTensorType outTensor =
            cast<RankedTensorType>(op.getValue().getType());
        
        auto outShape = outTensor.getShape();
        
        auto mv = dyn_cast<cliff::Cliff_MultivectorType>(outTensor.getElementType());
        auto activeComps = __builtin_popcountll(mv.getMask());

        auto llAttr =
            dyn_cast<LinearEncodingAttr>(outTensor.getEncoding());

        auto ll = llAttr.getLinearLayout();

        Value threadId =
            NVVM::ThreadIdXOp::create(rewriter, loc,
                                      rewriter.getI32Type());

        Value blockDim = NVVM::BlockDimXOp::create(rewriter, loc, rewriter.getI32Type());

        Value blockId =
            NVVM::BlockIdXOp::create(rewriter, loc,
                                     rewriter.getI32Type());

        auto kReg = StringAttr::get(ctx, "register");
        auto kLane = StringAttr::get(ctx, "lane");
        auto kWarp = StringAttr::get(ctx, "warp");
        auto kBlock = StringAttr::get(ctx, "block");

        Value c31 = b.i32_val(31);
        Value c5 = b.i32_val(5);
        Value zero = b.i32_val(0);

        bool moreThanOneWarp =
            (ll.getInDimSize(kWarp) > 1);

        Value regId = zero;

        Value laneId =
            moreThanOneWarp
                ? b.and_(threadId, c31)
                : threadId;

        Value warpId =
            moreThanOneWarp
                ? b.ashr_(threadId, c5)
                : zero;

        SmallVector<Value> inputCoords = {
            regId,
            laneId,
            warpId,
            blockId
        };

        // SmallVector<Value> logicalIndices =
        //     computeIndices(loc,
        //                    ll,
        //                    inputCoords,
        //                    rewriter);
        int32_t registerDims =
            std::max(ll.getInDimSize(kReg), 1);


        Value warpBase = b.add(
            b.add(
                b.mul(blockId, b.mul(blockDim, b.i32_val(registerDims))),
                b.mul(warpId, b.i32_val(32 * registerDims))
            ),
            laneId
        );
        
        assert((outShape[0] % 32 == 0) && "4now shapes should be multiples of 32");
        SmallVector<Value> globalRes;
        for (uint32_t regIdx = 0; regIdx < registerDims; ++regIdx) {
            Value regOffset = b.add(warpBase, b.mul(b.i32_val(regIdx), b.i32_val(32)));

            SmallVector<Value> res;
            for (uint32_t compIdx = 0; compIdx < activeComps; ++compIdx) {
                Value compOffset = b.add(regOffset, b.mul(b.i32_val(outShape[0]), b.i32_val(compIdx)));
                
                ValueRange values(compOffset);
                Value address =
                    b.gep(LLVM::LLVMPointerType::get(ctx),
                        rewriter.getF32Type(),
                        llPointer,
                        values);

                // todo: change once we allow more dtypes
                auto resTy = rewriter.getF32Type();
                Value compLoad = b.load(resTy, address);
                res.push_back(compLoad);
            }
            Value packedRes = packElements(loc, res, typeConverter, rewriter, mv);
            globalRes.push_back(packedRes);
        }
        
        Value result = packElements(loc, globalRes, typeConverter, rewriter, outTensor);
        
        rewriter.replaceOp(op, result);
        return success();
    }
};


class StoreOpPattern : public OpConversionPattern<StoreOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(StoreOp op,
                                  StoreOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {


        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();

        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        auto llPointer = adaptor.getPtr();

        auto typeConverter = getTypeConverter();
        RankedTensorType outTensor =
            cast<RankedTensorType>(op.getValue().getType());
        
        auto outShape = outTensor.getShape();
        
        auto mv = dyn_cast<cliff::Cliff_MultivectorType>(outTensor.getElementType());
        auto activeComps = __builtin_popcountll(mv.getMask());

        auto llAttr =
            dyn_cast<LinearEncodingAttr>(outTensor.getEncoding());

        auto ll = llAttr.getLinearLayout();
  
        Value threadId =
            NVVM::ThreadIdXOp::create(rewriter, loc,
                                      rewriter.getI32Type());

        Value blockId =
            NVVM::BlockIdXOp::create(rewriter, loc,
                                     rewriter.getI32Type());

        auto kReg = StringAttr::get(ctx, "register");
        auto kLane = StringAttr::get(ctx, "lane");
        auto kWarp = StringAttr::get(ctx, "warp");
        auto kBlock = StringAttr::get(ctx, "block");

        // compile-time known block dimension
        Value blockDim = NVVM::BlockDimXOp::create(rewriter, loc, rewriter.getI32Type());

        Value c31 = b.i32_val(31);
        Value c5 = b.i32_val(5);
        Value zero = b.i32_val(0);

        int32_t registerDims =
            std::max(ll.getInDimSize(kReg), 1);
        
        bool moreThanOneWarp =
            (ll.getInDimSize(kWarp) > 1);

        Value regId = zero;

        Value laneId =
            moreThanOneWarp
                ? b.and_(threadId, c31)
                : threadId;

        Value warpId =
            moreThanOneWarp
                ? b.ashr_(threadId, c5)
                : zero;

        SmallVector<Value> inputCoords = {
            regId,
            laneId,
            warpId,
            blockId
        };

        // i, j
        // SmallVector<Value> logicalIndices =
        //     computeIndices(loc,
        //                    ll,
        //                    inputCoords,
        //                    rewriter);

        Value tmp = b.mul(blockId, blockDim);
        Value idx = b.add(tmp, laneId);

        Value warpBase = b.add(
            b.add(
                b.mul(blockId, b.mul(blockDim, b.i32_val(registerDims))),
                b.mul(warpId, b.i32_val(32 * registerDims))
            ),
            laneId
        );

        Value llStruct = adaptor.getValue();
        SmallVector<Value> storeMvs = unpackElements(loc, llStruct, rewriter);
        
        for (uint32_t regIdx = 0; regIdx < registerDims; ++regIdx) {
            SmallVector<Value> storeComps = unpackElements(loc, storeMvs[regIdx], rewriter);
            Value regOffset = b.add(warpBase, b.mul(b.i32_val(regIdx), b.i32_val(32)));
            for (uint32_t i = 0; i < storeComps.size(); ++i) {
                Value compOffset = b.add(regOffset, b.mul(b.i32_val(outShape[0]), b.i32_val(i)));
                ValueRange values(compOffset);
                Value address =
                    b.gep(LLVM::LLVMPointerType::get(ctx),
                        rewriter.getF32Type(),
                        llPointer,
                        values);

                // todo: change once we allow more dtypes
                auto resTy = rewriter.getF32Type();
                b.store(storeComps[i], address);
            }
        }
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace 

void mlir::cliff::populateLoadStoreOpPatterns(CliffGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
    MLIRContext *context = patterns.getContext();
    patterns.insert<LoadOpPattern, StoreOpPattern>(typeConverter, context);
}