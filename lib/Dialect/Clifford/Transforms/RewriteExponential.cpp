#include "clifford/Dialect/Clifford/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "rewrite-exponential"

namespace mlir::cliff {
    #define GEN_PASS_DEF_REWRITEEXPONENTIALPASS
    #include "clifford/Dialect/Clifford/Transforms/CliffPasses.h.inc"

} // end namespace mlir::cliff

using namespace mlir;
using namespace mlir::cliff;

class RewriteExponentialPattern : public OpRewritePattern<Exp> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(Exp op, PatternRewriter &rewriter) const override {
        LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] Trying to match: " << op << "\n");

        RankedTensorType srcTensor = dyn_cast<RankedTensorType>(op.getSrc().getType());
        RankedTensorType outTensor = dyn_cast<RankedTensorType>(op.getResult().getType());

        if (!srcTensor) {
            LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] FAIL: src is not a RankedTensorType\n");
            return failure();
        }

        auto srcMultivector = dyn_cast<Cliff_AlgebraicElementInterface>(srcTensor.getElementType());
        if (!srcMultivector) {
            LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] FAIL: element type is not a GeometricObject, got: "
                                    << srcTensor.getElementType() << "\n");
            return failure();
        }

        auto space = dyn_cast<CliffordAlgebraAttr>(srcMultivector.getAlgebra());
        auto p = space.getP(), q = space.getQ(), r = space.getR();
        LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] Algebra: p=" << p << " q=" << q << " r=" << r << "\n");
        
        if (q != 0 || r != 1 || (p < 1))
            return rewriter.notifyMatchFailure(op, "This pass only supports PGA");

        GeoProd geoProd = dyn_cast<GeoProd>(op.getSrc().getDefiningOp());
        if (!geoProd) {
            LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] FAIL: src is not defined by a GeoProd\n");
            return failure();
        }

        auto scalarTensor = dyn_cast<RankedTensorType>(geoProd.getLhs().getType());
        if (!dyn_cast<Cliff_ScalarType>(scalarTensor.getElementType()))
            return failure();
        
        auto axisTensor = dyn_cast<RankedTensorType>(geoProd.getRhs().getType());
        if (!axisTensor)
            return failure();

        LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] GeoProd lhs type: " << geoProd.getLhs().getType() << "\n");
        LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] GeoProd rhs type: " << geoProd.getRhs().getType() << "\n");
        LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] GeoProd result type: " << geoProd.getResult().getType() << "\n");

        if (!dyn_cast<Cliff_MotorType>(outTensor.getElementType())) {
            LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] FAIL: result element type is not a MotorType, got: "
                                    << outTensor.getElementType() << "\n");
            return failure();
        }

        auto geoObject = dyn_cast<Cliff_GeometricElementInterface>(axisTensor.getElementType());
        if (!geoObject) {
            LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] FAIL: referenceLocusTensor element type is not a Cliff_GeometricElementInterface, got: "
                                    << axisTensor.getElementType() << "\n");
            return failure();
        }

        auto referenceLocus = cast<Cliff_MultivectorType>(geoObject.asMultivector());
        // todo : strictly more conditions must be met (B is simple, B^2 is scalar)
        if (referenceLocus.getDegree() == 2) {

            if (geoObject.getObjectKind().getValue() == ObjectKind::Euclidean) {
                LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] -> Rotate (Euclidean referenceLocus)\n");
                bool hasOneUse = geoProd.getResult().hasOneUse();
                rewriter.replaceOpWithNewOp<Rotate>(op, outTensor, geoProd.getLhs(), geoProd.getRhs());
                if (hasOneUse)
                    rewriter.eraseOp(geoProd);
                return success();
            }
            if (geoObject.getObjectKind().getValue() == ObjectKind::Ideal) {
                LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] -> Translate (Ideal referenceLocus)\n");
                bool hasOneUse = geoProd.getResult().hasOneUse();
                rewriter.replaceOpWithNewOp<Translate>(op, outTensor, geoProd.getLhs(), geoProd.getRhs());
                if (hasOneUse)
                    rewriter.eraseOp(geoProd);
                return success();
            }
        }
        LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] FAIL: referenceLocus degree is not 2, got: "
                                    << referenceLocus.getDegree() << "\n");
        LLVM_DEBUG(llvm::dbgs() << "[RewriteExp] FAIL: no pattern matched\n");
        return failure();
    }
};


class RewriteExponentialPass : public cliff::impl::RewriteExponentialPassBase<RewriteExponentialPass> {
public:
    using RewriteExponentialPassBase::RewriteExponentialPassBase;
        
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        auto *mod = getOperation();
        patterns.insert<RewriteExponentialPattern>(context);

        if (applyPatternsGreedily(mod, std::move(patterns)).failed())
            signalPassFailure();
        
    }

};