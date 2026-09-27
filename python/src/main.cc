#include <nanobind/nanobind.h>

namespace py = nanobind;

void init_clifford_ir(py::module_ &m);

NB_MODULE(libclifford, m) {
    
    auto ir_m = m.def_submodule("ir");
    init_clifford_ir(ir_m);

}
