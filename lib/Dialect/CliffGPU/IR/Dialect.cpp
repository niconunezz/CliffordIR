#include "clifford/Dialect/CliffGPU/IR/Dialect.h"

#include "clifford/Dialect/CliffGPU/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "clifford/Dialect/CliffGPU/IR/CliffGPUAttrDefs.cpp.inc"
#undef GET_ATTRDEF_CLASSES

using namespace mlir::clg;
using namespace mlir::cliff;

using namespace mlir;
using namespace llvm;




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


void CliffGPUDialect::initialize() {


    addAttributes<
        #define GET_ATTRDEF_LIST
        #include "clifford/Dialect/CliffGPU/IR/CliffGPUAttrDefs.cpp.inc"
        
    >();



}