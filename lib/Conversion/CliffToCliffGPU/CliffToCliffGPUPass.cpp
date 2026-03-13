#include "clifford/Dialect/CliffGPU/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "clifford/Conversion/CliffToCliffGPU/Passes.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"


namespace mlir::cliff {
    #define GEN_PASS_DEF_CONVERTCLIFFTOCLIFFGPU
    #include "clifford/Conversion/CliffToCliffGPU/Passes.h.inc"

}


namespace {

using namespace mlir;
using namespace mlir::cliff;
using namespace mlir::clg;


struct 


}