#include <nanobind/nanobind.h>

namespace py = nanobind;

void init_clifford_ir(py::module_ &m);
void init_clifford_passes(py::module_ &m);
void init_llvm_ir(py::module_ &m);
void init_driver(py::module_ &m);

NB_MODULE(libclifford, m) {
    
    auto ir_m = m.def_submodule("ir");
    init_clifford_ir(ir_m);

    auto passes_m = m.def_submodule("passes");
    init_clifford_passes(passes_m);

    auto llvm_m = m.def_submodule("llvm");
    init_llvm_ir(llvm_m);

    auto driver_m = m.def_submodule("driver");
    init_driver(driver_m);
    
}
