#ifndef INCLUDE_CLIFFORD_TOOLS_STRUTILS_H
#define INCLUDE_CLIFFORD_TOOLS_STRUTILS_H

#include <string>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"


namespace mlir::cliff {


template <typename C>
std::string join(C &&container, llvm::StringRef sep) {
    std::string ret;
    llvm::raw_string_ostream s(ret);

    for (auto const &elem : container) {
        if (!ret.empty())
            s << sep; 
    
        s << elem;
    }
    return ret;
}


template <typename C, typename Fn>
std::string join(C &&container, llvm::StringRef sep, Fn &&fn) {
    std::string ret;
    llvm::raw_string_ostream s(ret);

    for (auto const &elem : container) {
        if (!ret.empty())
            s << sep;
        
        if constexpr (std::is_invocable_v<Fn, llvm::raw_ostream &, decltype(elem)>) {
            static_assert( std::is_void_v<std::invoke_result_t<Fn, llvm::raw_ostream &, decltype(elem)>>);
            fn(s, elem);
        }
        else {
            s << fn(elem);
        }
    }

    return ret;
}

} // namespace mlir::cliff


#endif // INCLUDE_CLIFFORD_TOOLS_STRUTILS_H