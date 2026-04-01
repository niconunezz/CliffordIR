#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"

#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

namespace mlir::clg {
#define GEN_PASS_DEF_CONVERTCLIFFGPUTOLLVM
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h.inc"

} // namespace mlir::clg



namespace {

using namespace mlir;
using namespace mlir::clg;


class ConvertCliffGPUToLLVM : public clg::impl::ConvertCliffGPUToLLVMBase<ConvertCliffGPUToLLVM> {
public:
using ConvertCliffGPUToLLVMBase::ConvertCliffGPUToLLVMBase;

    void runOnOperation() override {

    }

};

}