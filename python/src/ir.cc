#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"

namespace py = nanobind;
using namespace mlir;
using namespace cliff;

class CliffordOpBuilder {  
public:
    CliffordOpBuilder(MLIRContext* ctx) {
        builder = std::make_unique<OpBuilder>(ctx);
        lastLoc = std::make_unique<Location>(builder->getUnknownLoc());
    }

    template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
        auto loc = getLastLoc();
        return OpTy::create(*builder, loc, std::forward<Args>(args)...);
    }

    void setLastLoc(Location loc) {
        lastLoc = std::make_unique<Location>(loc);
    }

    void setInsertionPointToStart(Block *block) {
        if (block->isEntryBlock())
            setLastLoc(block->begin()->getLoc());

        builder->setInsertionPointToStart(block);
    }

    void setInsertionPointAfter(Operation *op) {
        setLastLoc(op->getLoc());
        builder->setInsertionPointAfter(op);
    }

    OpBuilder &getBuilder() {
        return *builder;
    }

    Location getLastLoc() {
        return *lastLoc;
    }

private:
    std::unique_ptr<OpBuilder> builder;
    std::unique_ptr<Location> lastLoc;
};

void init_clifford_ir(py::module_ &m) {

    py::class_<MLIRContext>(m, "context")
    .def(py::new_([]() {return new MLIRContext(MLIRContext::Threading::DISABLED);}));

    py::class_<OpState>(m, "OpState")
    .def("dump", [](OpState &self) { self->dump(); });

    py::class_<ModuleOp, OpState>(m, "module")
    .def("dump", &ModuleOp::dump)
    .def("push_back", [](ModuleOp &self, func::FuncOp &funcOp) {
        self.push_back(funcOp);
    });

    py::class_<FunctionType>(m, "FunctionType");

    
    py::class_<FuncOp, OpState>(m, "FuncOp")
    .def("addEntryBlock", &FuncOp::addEntryBlock)
    .def("args",
        [](FuncOp &self, unsigned idx) -> BlockArgument {
            if (idx >= self.getNumArguments())
            throw py::index_error("Function argument index out of range");
            return self.getArgument(idx);
        })
        .def("get_num_args", &FuncOp::getNumArguments);
        
    py::class_<mlir::FloatType>(m, "FloatType");

    py::class_<CliffordOpBuilder>(m, "CliffordOpBuilder")
    .def(py::init<MLIRContext *>())
    .def("set_insertion_point_to_start", [](CliffordOpBuilder &self, Block *block) {
        self.setInsertionPointToStart(block);
    })
    .def("get_function_type", [](CliffordOpBuilder &self, std::vector<Type> ins, std::vector<Type> outs) -> Type {
        return self.getBuilder().getFunctionType(ins, outs);
    })
    .def("get_f32ty", [](CliffordOpBuilder &self) {
        return self.getBuilder().getF32Type();
    })
    .def("create_rotate", [](CliffordOpBuilder &self, Value angle, Value src) {
        return self.create<Rotate>(self.getBuilder().getF32Type(), angle, src);
    })
    .def("create_module", [](CliffordOpBuilder &self) {
        return self.create<ModuleOp>();
    })
    .def("get_function_def", [](CliffordOpBuilder &self, std::string &name, Type &fnTy, std::string &visibility) -> FuncOp {
        if (auto funcTy = dyn_cast<FunctionType>(fnTy)) {
            StringAttr fnNameAttr = self.getBuilder().getStringAttr(name);
            TypeAttr funcTyAttr = TypeAttr::get(funcTy);
            StringAttr visibilityAttr = self.getBuilder().getStringAttr(visibility);
            mlir::ArrayAttr argAttrs = {};
            mlir::ArrayAttr resAttrs = {};

            return self.create<FuncOp>(fnNameAttr, funcTyAttr, visibilityAttr, argAttrs, resAttrs);
        }      
        throw std::invalid_argument("invalid function type");
    },
    py::arg("fn_name"), py::arg("fn_type"), py::arg("visibility"));
}
