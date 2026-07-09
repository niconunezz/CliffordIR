#include "clifford/Dialect/Clifford/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "rewrite-sandwich"

namespace mlir::cliff {
    #define GEN_PASS_DEF_REWRITESANDWICHPASS
    #include "clifford/Dialect/Clifford/Transforms/CliffPasses.h.inc"

} // end namespace mlir::cliff

using namespace mlir;
using namespace mlir::cliff;

class RewriteSandwichPattern : public OpRewritePattern<Sandwich> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(Sandwich op, PatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] Trying to match: " << op << "\n");

        RankedTensorType motorTensor = dyn_cast<RankedTensorType>(op.getLhs().getType());
        RankedTensorType objTensor = dyn_cast<RankedTensorType>(op.getRhs().getType());

        RankedTensorType outTensor = dyn_cast<RankedTensorType>(op.getResult().getType());

        if (!motorTensor || !objTensor) {
            LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] FAIL: motor or obj are not a RankedTensorType\n");
            return failure();
        }

        auto motor = dyn_cast<Cliff_MotorType>(motorTensor.getElementType());
        auto obj = dyn_cast<Cliff_GeometricElementInterface>(objTensor.getElementType());
        if (!motor || !obj) {
            LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] FAIL: element type is not a what is supposed to be, got: "
                                    << motorTensor.getElementType() << " and " << objTensor.getElementType() << "\n");
            return failure();
        }

        auto space = dyn_cast<CliffordAlgebraAttr>(motor.getAlgebra());
        auto p = space.getP(), q = space.getQ(), r = space.getR();
        LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] Algebra: p=" << p << " q=" << q << " r=" << r << "\n");
        
        if (q != 0 || r != 1 || (p < 1))
            return failure();

        MLIRContext *ctx = op.getContext();
        auto resultMask = getResultMask(motor.getActiveMask(), obj.getActiveMask(), p, q, r);
        LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] Computed first result mask : " << resultMask << "\n");

        Cliff_MultivectorType mvType = Cliff_MultivectorType::get(ctx, resultMask,
                                                                  rewriter.getF32Type(), 
                                                                  GeometricKindAttr::get(ctx, GeometricKind::Unknown),
                                                                  space);
        LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] Obtained its mvType" << "\n");
        
        Value geoProdLhs = GeoProd::create(rewriter, loc, RankedTensorType::get(motorTensor.getShape(), mvType), op.getLhs(), op.getRhs());
        LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] Created first geoProd" << "\n");

        //todo : think about this
        bool normalized = false;
        Value motorRev = Reverse::create(rewriter, loc,
             RankedTensorType::get(motorTensor.getShape(), motor), op.getLhs());
        LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] Created reversed" << "\n");
        Cliff_PointType pointTy = Cliff_PointType::get(ctx, obj.getObjectKind(), false, space);
        Value geoProdRhs = GeoProd::create(rewriter, loc, RankedTensorType::get(motorTensor.getShape(), outTensor.getElementType()), geoProdLhs, motorRev);
        LLVM_DEBUG(llvm::dbgs() << "[RewriteSandwich] Created second geoProd" << "\n");

        rewriter.replaceOp(op, geoProdRhs);
        return success();
    }
};


class RewriteSandwichPass : public cliff::impl::RewriteSandwichPassBase<RewriteSandwichPass> {
public:
    using RewriteSandwichPassBase::RewriteSandwichPassBase;
        
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        auto *mod = getOperation();
        patterns.insert<RewriteSandwichPattern>(context);

        if (applyPatternsGreedily(mod, std::move(patterns)).failed())
            signalPassFailure();
        
    }

};