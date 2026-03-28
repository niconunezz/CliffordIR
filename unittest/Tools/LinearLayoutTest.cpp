#include "clifford/Tools/LinearLayout.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// build with cmake --build . --target LinearLayout && 
// run with ./unittest/Tools/LinearLayout

namespace mlir::clg {
namespace {

using ::llvm::to_vector;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

using BasesT = LinearLayout::BasesT;


class LinearLayoutTest : public ::testing::Test {
public:
    StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
    MLIRContext ctx;
};

TEST_F(LinearLayoutTest, Empty) {
  LinearLayout layout = LinearLayout::empty();
  EXPECT_THAT(layout.getBases(), IsEmpty());
  EXPECT_THAT(to_vector(layout.getInDimNames()), IsEmpty());
  EXPECT_THAT(to_vector(layout.getOutDimNames()), IsEmpty());
}






} // namespace
} // namespace mlir::cliff

int main(int argc, char *argv[]) {
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}