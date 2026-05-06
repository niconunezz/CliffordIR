#include "mlir/Pass/Pass.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Conversion/CliffGPUToLLVM/PopulatePatterns.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"

using namespace mlir;
using namespace mlir::cliff;


namespace {
class RewriteRotatePattern : public OpRewritePattern<Rotate> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(Rotate op,
                                  PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto fpTy = rewriter.getF32Type();
        
        Value cos = math::CosOp::create(rewriter, loc, fpTy, op.getAngle());
        Value sin = math::SinOp::create(rewriter, loc, fpTy, op.getAngle());
        Value geoProd = GeoProd::create(rewriter, loc, fpTy, sin, op.getSrc());
        Value ret = arith::AddFOp::create(rewriter, loc, cos, geoProd);

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
        
        Value one = arith::ConstantOp::create(rewriter, loc,
                    rewriter.getFloatAttr(fpTy, 1.0));
        Value scaledMv = GeoProd::create(rewriter, loc, fpTy, op.getDist(), op.getSrc());
        Value ret = arith::AddFOp::create(rewriter, loc, one, scaledMv);

        rewriter.replaceOp(op, ret);
        return success();
    }

};
} // namespace

void mlir::cliff::populateGeometricRewritePatterns(RewritePatternSet &patterns) {
    MLIRContext* context = patterns.getContext();
    patterns.insert<RewriteRotatePattern, RewriteTranslatePattern>(context);
}