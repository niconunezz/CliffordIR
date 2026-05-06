
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"

#include "clifford/Dialect/CliffGPU/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "clifford/Dialect/CliffGPU/IR/CliffGPUAttrDefs.cpp.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_OP_CLASSES
#include "clifford/Dialect/CliffGPU/IR/CliffGPUOps.cpp.inc"

using namespace mlir::clg;
using namespace mlir::cliff;

using namespace mlir;
using namespace llvm;

namespace mlir::clg {

SmallVector<StringAttr> getStandardOutDims(MLIRContext *ctx, unsigned N) {
    SmallVector<StringAttr> ret;
    for (int i = 0; i < N; ++i) {
        ret.emplace_back(StringAttr::get(ctx, llvm::Twine("dim") + llvm::Twine(i)));
    }
    return ret;
}
unsigned LinearEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
    const auto &ll = getLinearLayout();

    StringAttr kReg = StringAttr::get(getContext(), "register");
    return ll.getInDimSize(kReg);
}


unsigned LinearEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape) const {
    return getElemsPerThread(shape);
}


LinearEncodingAttr getDefaultGlobalEncoding(MLIRContext *ctx, int32_t numWarps, int32_t threadsPerWarp, ArrayRef<int64_t> shape) {

    unsigned rank = shape.size();
    for (uint16_t i=0; i < rank; ++i)
        assert(shape[i] >= 32 && "Tensor shape must be at least 32 for each dimension");
    SmallVector<unsigned> order(rank);
    std::iota(order.begin(), order.end(), 0);
    std::reverse(order.begin(), order.end());

    StringAttr kReg = StringAttr::get(ctx, "register");
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");

    ArrayRef<StringAttr> outDims = getStandardOutDims(ctx, rank);
    
    //todo : take this out, [] should be allowed
    auto ll = LinearLayout::zeros1D(2, kReg, outDims[0]); // we ensure theres always a reg dimension
    
    //todo : generalize this to any number of dimensions
    unsigned warpsPerRow = shape[order[0]] / threadsPerWarp;
    
    ll *= LinearLayout::identity1D(threadsPerWarp, kLane, outDims[0]);
    ll *= LinearLayout::identity1D(warpsPerRow, kWarp, outDims[0]);
    
    if (rank == 2) {
        unsigned warpsPerCol = numWarps/warpsPerRow;
        unsigned numBlocks = shape[0] / warpsPerCol;
    
        ll *= LinearLayout::identity1D(warpsPerCol, kWarp, outDims[1]);
        ll *= LinearLayout::identity1D(numBlocks, kBlock, outDims[1]);

    }

    return LinearEncodingAttr::get(ctx, ll);
}


std::optional<LinearLayout> parseLinearLayout(AsmParser& parser, 
                                              const DictionaryAttr &dict, 
                                              const std::vector<std::string> &inDimNames) {
    

    LinearLayout::BasesT bases;

    for (const auto &inDimNameStr : inDimNames) {
        auto inDimName = StringAttr::get(parser.getContext(), inDimNameStr);
        Attribute value = dict.get(inDimName);

        if (!value) {
            parser.emitError(parser.getCurrentLocation(), "Expected basis of '")
            << inDimName.getValue() << "' not found";
            return {};
        }
        
        auto arrayOfArraysAttr = dyn_cast<ArrayAttr>(value);
        if (!arrayOfArraysAttr) {
            parser.emitError(parser.getCurrentLocation(), "Expected array of arrays for basis of '")
            << inDimName.getValue() << "'";
            return {};
        }

        std::vector<std::vector<int32_t>> inDimBases;
        for (Attribute arrayAttr : arrayOfArraysAttr) {
            auto intArrayAttr = dyn_cast<ArrayAttr>(arrayAttr);
            if (!intArrayAttr) {
                parser.emitError(parser.getCurrentLocation(), "Expected array for basis of '")
                << inDimName.getValue() << "'"; 
                return {};
            }
            std::vector<int32_t> basis;
            for (Attribute intAttr : intArrayAttr) {
                auto intAttrVal = dyn_cast<IntegerAttr>(intAttr);
                if (!intAttrVal) {
                    parser.emitError(parser.getCurrentLocation(), "Expected integer for basis of '")
                    << inDimName.getValue() << "'";
                    return {};
                }
                basis.push_back(intAttrVal.getInt());
            }
            inDimBases.push_back(std::move(basis));
        }

        bases[inDimName] = std::move(inDimBases);

    }
    int rank = 0;
    
    for (const auto &basesDim : llvm::make_second_range(bases)) {
        if (!basesDim.empty()) {
            rank = basesDim[0].size();
            break;
        }
    }
    SmallVector<StringAttr> outDimNames;
    for (int i = 0; i < rank; i++) {
        outDimNames.push_back(
            StringAttr::get(parser.getContext(), "dim" + llvm::Twine(i)));
    }
    return LinearLayout(std::move(bases), std::move(outDimNames));

}


Attribute LinearEncodingAttr::parse(AsmParser& parser, Type type) {

    if (parser.parseLess().failed()) 
        return {};
    
    DictionaryAttr dict;
    if (parser.parseAttribute(dict).failed()) 
        return {};
    
    if (parser.parseGreater().failed()) 
        return {};
    

    std::vector<std::string> inDimNames = {"register", "lane", "warp", "block"};
    auto maybeLinearLayout = parseLinearLayout(parser, dict, inDimNames);

    if (!maybeLinearLayout.has_value())
        return {};

    return parser.getChecked<LinearEncodingAttr>(parser.getContext(), std::move(*maybeLinearLayout));
}
void printLinearLayout(AsmPrinter &printer, const LinearLayout &ll, bool skipEmptyBases = false) {

    auto bases = ll.getBases();
    if (skipEmptyBases) {

        decltype(bases) filtered;
        for (auto const &kv : bases) 
            if (!kv.second.empty())
                filtered.insert(kv);
        
        bases = std::move(filtered);
    }

    printer << join(bases, ", ", [](const auto &base) {
        return base.first.str() + " = " + "[ " + join(base.second, ", ", [](const std::vector<int32_t> &vec){
            return "[" + join(vec, ", ") + "]";
        }) + "]";
    });
}

void LinearEncodingAttr::print(AsmPrinter& printer) const {
    printer << "<{";
    printLinearLayout(printer, getLinearLayout());
    printer << "}>";

}


LogicalResult LinearEncodingAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout) {
    //todo : fill;
    return success();
}


struct CliffGPUOpAsmDialectInterface : public OpAsmDialectInterface {
    using OpAsmDialectInterface::OpAsmDialectInterface;

    AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
        if (isa<LinearEncodingAttr>(attr)) {
            os << "layout";
            return AliasResult::FinalAlias;
        }
        return AliasResult::NoAlias;
    }
};

} // namespace mlir::clg

void CliffGPUDialect::initialize() {


    addAttributes<
        #define GET_ATTRDEF_LIST
        #include "clifford/Dialect/CliffGPU/IR/CliffGPUAttrDefs.cpp.inc"
        
    >();

    addInterfaces<CliffGPUOpAsmDialectInterface>();

    addOperations<
        #define GET_OP_LIST
        #include "clifford/Dialect/CliffGPU/IR/CliffGPUOps.cpp.inc"
        >();

}


