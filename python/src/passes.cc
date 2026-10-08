#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlir/Pass/PassManager.h"
#include "clifford/Dialect/Clifford/Transforms/Passes.h"
#include "clifford/Conversion/CliffToCliffGPU/Passes.h"
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"


using namespace mlir;
namespace py = nanobind;


void cliff_passes(py::module_ &m) {
    m.def("create_rewrite_exponential_pass", [](mlir::PassManager &pm) { pm.addPass(cliff::createRewriteExponentialPass()); });
    m.def("create_rewrite_sandwich_pass", [](mlir::PassManager &pm) { pm.addPass(cliff::createRewriteSandwichPass()); });
    m.def("create_geometric_type_conversion_pass", [](mlir::PassManager &pm) { pm.addPass(cliff::createGeometricTypeConversionPass()); });
}

void cliff_to_clg(py::module_ &m) {
    m.def("create_cliff_to_cliffgpu", [](mlir::PassManager &pm) { pm.addPass(cliff::createConvertCliffToCliffGPU()); });
}

void clg_to_llvm(py::module_ &m) {
    m.def("create_cliffgpu_to_llvm", [](mlir::PassManager &pm) { pm.addPass(clg::createConvertCliffGPUToLLVM()); });
}

void init_clifford_passes(py::module_ &m) {
    auto cliff_passes_m = m.def_submodule("cliff_passes");
    cliff_passes(cliff_passes_m);

    auto cliff_to_clg_m = m.def_submodule("cliff_to_clg");
    cliff_to_clg(cliff_to_clg_m);

    auto clg_to_llvm_m = m.def_submodule("llvm");
    clg_to_llvm(clg_to_llvm_m);

}