#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
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

    void setLastLoc(const std::string &fileName, int line, int column) {
        auto context = builder->getContext();
        setLastLoc(mlir::FileLineColLoc::get(context, fileName, line, column));
    }

    void setInsertionPointToStart(Block &block) {
        if (!block.empty()){
            setLastLoc(block.begin()->getLoc());
        }

        builder->setInsertionPointToStart(&block);
    }

    void setInsertionPointAfter(Operation &op) {
        setLastLoc(op.getLoc());
        builder->setInsertionPointAfter(&op);
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
    using ret = py::rv_policy;

    py::class_<MLIRContext>(m, "context")
    .def(py::new_([]() {return new MLIRContext(MLIRContext::Threading::DISABLED);}));

    py::class_<OpState>(m, "OpState")
    .def("dump", [](OpState &self) { self->dump(); });

    py::class_<ModuleOp, OpState>(m, "module")
    .def("dump", &ModuleOp::dump)
    .def("push_back", [](ModuleOp &self, FuncOp &funcOp) {
        self.push_back(funcOp);
    })
    .def("get_body", [](ModuleOp &self) { return self.getBody(); }, ret::reference);

    py::class_<FunctionType>(m, "FunctionType");

    py::class_<Block>(m, "Block")
    .def("get_parent", &Block::getParent, ret::reference)
    .def("dump", &Block::dump);
    
    py::class_<Region>(m, "Region")
    .def("get_context", &Region::getContext, ret::reference)
    .def("get_location", &Region::getLoc)
    .def("get_argument_types", &Region::getArgumentTypes)
    .def("front", &Region::front, ret::reference);

    py::class_<FuncOp, OpState>(m, "FuncOp")
    .def("add_entry_block", &FuncOp::addEntryBlock, ret::reference)
    .def("args",
        [](FuncOp &self, unsigned idx) -> BlockArgument {
            if (idx >= self.getNumArguments())
            throw py::index_error("Function argument index out of range");
            return self.getArgument(idx);
        })
    .def("get_body",[](FuncOp &self) -> Region& {
        return self.getBody();
    }, ret::reference)
    .def("get_num_args", &FuncOp::getNumArguments)
    .def("debug", [](FuncOp &self) {
        self.getFunctionType().dump();
        if (!self.getBody().empty()) {
            Block &b = self.getBody().front();
            b.dump();
        }
    });
        
    py::class_<mlir::FloatType>(m, "FloatType");

    py::class_<mlir::Type>(m, "Type")
    .def("__repr__", [](mlir::Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return str;
    });
    
    py::class_<Value>(m, "Value");

    py::class_<Operation>(m, "Operation")
    .def("get_name", &Operation::getName);

    py::class_<CliffordOpBuilder>(m, "CliffordOpBuilder")
    .def(py::init<MLIRContext *>())
    .def("set_insertion_point_to_start", [](CliffordOpBuilder &self, Block &block) {
        self.setInsertionPointToStart(block);
    })
    .def("get_function_type", [](CliffordOpBuilder &self, std::vector<Type> ins, std::vector<Type> outs) -> Type {
        return self.getBuilder().getFunctionType(TypeRange(ins), TypeRange(outs));
    })
    .def("get_f32ty", [](CliffordOpBuilder &self) -> Type {
        return self.getBuilder().getF32Type();
    })
    .def("create_rotate", [](CliffordOpBuilder &self, Value angle, Value src) -> OpState {
        return self.create<Rotate>(self.getBuilder().getF32Type(), angle, src);
    })
    .def("create_module", [](CliffordOpBuilder &self) {
        return self.create<ModuleOp>();
    })
    .def("create_return", [](CliffordOpBuilder &self, std::vector<Value> outTys) -> OpState {
        return self.create<ReturnOp>(mlir::ValueRange(outTys));
    })
    .def("set_location",  [](CliffordOpBuilder &self, std::string fileName, int line, int column) -> void {
        return self.setLastLoc(fileName, line, column);
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

    m.def("register_dialects", [](MLIRContext *ctx) {
        mlir::DialectRegistry registry;

        registry.insert<CliffDialect>();

        ctx->appendDialectRegistry(registry);
        ctx->loadAllAvailableDialects();
    });
}
