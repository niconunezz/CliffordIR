#include "mlir/Pass/Pass.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Conversion/CliffGPUToLLVM/PopulatePatterns.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"

using namespace mlir;
using namespace mlir::cliff;
using namespace mlir::clg;


namespace {
class RewriteRotatePattern : public OpRewritePattern<Rotate> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(Rotate op,
                                  PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto fpTy = rewriter.getF32Type();
        auto b = CliffordLLVMOpBuilder(loc, rewriter);

        auto cos = b.cos(fpTy, op.getAngle());
        cos.setFastmathFlags(LLVM::FastmathFlags::afn);

        auto sin = b.sin(fpTy, op.getAngle());
        sin.setFastmathFlags(LLVM::FastmathFlags::afn);

        Value geoProd = GeoProd::create(rewriter, loc, fpTy, sin, op.getSrc());
        Value ret = b.fadd(cos, geoProd);

        rewriter.replaceOp(op, ret);
        return success();
    }

};


class RewriteTranslatePattern : public OpRewritePattern<Translate> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(Translate op,
                                  PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto fpTy = rewriter.getF32Type();
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        Value one = b.f32_val(1.0);
        Value scaledMv = GeoProd::create(rewriter, loc, fpTy, op.getDist(), op.getSrc());
        Value ret = b.fadd(one, scaledMv);

        rewriter.replaceOp(op, ret);
        return success();
    }

};
} // namespace

void mlir::cliff::populateGeometricRewritePatterns(RewritePatternSet &patterns) {
    MLIRContext* context = patterns.getContext();
    patterns.insert<RewriteRotatePattern, RewriteTranslatePattern>(context);
}