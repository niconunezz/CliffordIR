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


TEST_F(LinearLayoutTest, Identity1D) {
  LinearLayout layout =
      LinearLayout::identity1D(32, S("testIns"), S("testOuts")); 
  SmallVector<StringAttr> outDims = {S("testOuts")};
  LinearLayout layout2 = LinearLayout({{S("testIns"), {{1}, {2}, {4}, {8}, {16}}}}, outDims);

  EXPECT_THAT(layout, layout2);
  EXPECT_THAT(to_vector(layout.getInDimNames()), ElementsAre(S("testIns")));
  EXPECT_THAT(to_vector(layout.getOutDimNames()), ElementsAre(S("testOuts")));
  EXPECT_THAT(layout.getInDimSizeLog2(S("testIns")), 5);
  EXPECT_THAT(layout.getOutDimSizeLog2(S("testOuts")), 5);
}

TEST_F(LinearLayoutTest, Identity1DSize1) {
  LinearLayout layout =
      LinearLayout::identity1D(1, S("testIns"), S("testOuts"));
  EXPECT_EQ(layout, LinearLayout({{S("testIns"), {}}}, {S("testOuts")}));
  EXPECT_THAT(to_vector(layout.getInDimNames()), ElementsAre(S("testIns")));
  EXPECT_THAT(to_vector(layout.getOutDimNames()), ElementsAre(S("testOuts")));
  EXPECT_THAT(layout.getInDimSizeLog2(S("testIns")), 0);
  EXPECT_THAT(layout.getOutDimSizeLog2(S("testOuts")), 0);
}


TEST_F(LinearLayoutTest, MultiplyIdentity) {
  LinearLayout prod = LinearLayout::identity1D(16, S("in"), S("out")) *
                      LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout(
                      {{S("in"),
                        {{1}, {2}, {4}, {8}, {16}, {32}, {64}, {128}, {256}}}},
                      {S("out")}));
  EXPECT_THAT(to_vector(prod.getInDimNames()), ElementsAre(S("in")));
  EXPECT_THAT(to_vector(prod.getOutDimNames()), ElementsAre(S("out")));
}

TEST_F(LinearLayoutTest, MultiplyDisjoint) {
  LinearLayout prod = LinearLayout::identity1D(32, S("in1"), S("out1")) *
                      LinearLayout::identity1D(16, S("in2"), S("out2"));
  EXPECT_EQ(prod, LinearLayout(
                      {
                          {S("in1"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                          {S("in2"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                      },
                      {S("out1"), S("out2")}));
  EXPECT_THAT(to_vector(prod.getInDimNames()), ElementsAre(S("in1"), S("in2")));
  EXPECT_THAT(to_vector(prod.getOutDimNames()),
              ElementsAre(S("out1"), S("out2")));
}

TEST_F(LinearLayoutTest, MultiplyOverlapping) {
  LinearLayout prod = LinearLayout::identity1D(4, S("in"), S("out1")) *
                      LinearLayout::identity1D(8, S("in"), S("out2"));
  EXPECT_EQ(prod,
            LinearLayout({{S("in"), {{1, 0}, {2, 0}, {0, 1}, {0, 2}, {0, 4}}}},
                         {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, TimesEquals) {
  LinearLayout prod = LinearLayout::empty();
  prod *= LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout::identity1D(32, S("in"), S("out")));
}

TEST_F(LinearLayoutTest, GetOutDimSizeLog2) {
  LinearLayout layout(
      {
          {S("in0"), {{0}, {0}, {0}}},
          {S("in1"), {{1}, {2}}},
      },
      {S("dim0")});
  EXPECT_EQ(layout.getOutDimSizeLog2(S("dim0")), 2);
}

TEST_F(LinearLayoutTest, Apply) {
  LinearLayout layout(
      {
          {S("in1"), {{4, 2}, {2, 1}, {1, 0}}},
          {S("in2"), {{1, 2}, {2, 1}}},
      },
      {{S("out1"), 8}, {S("out2"), 4}});
  EXPECT_THAT(layout.apply({{S("in1"), 0}, {S("in2"), 0}}),
              ElementsAre(Pair(S("out1"), 0), Pair(S("out2"), 0)));
  EXPECT_THAT(layout.apply({{S("in2"), 0}, {S("in1"), 1}}),
              ElementsAre(Pair(S("out1"), 4), Pair(S("out2"), 2)));
  EXPECT_THAT(layout.apply({{S("in2"), 1}, {S("in1"), 0}}),
              ElementsAre(Pair(S("out1"), 1), Pair(S("out2"), 2)));
}

TEST_F(LinearLayoutTest, Sublayout) {
  LinearLayout l1({{S("in1"), {{1, 0}, {0, 1}, {2, 0}}}, {S("in2"), {{0, 1}}}},
                  {S("out1"), S("out2")});
  EXPECT_EQ(l1.sublayout({S("in1"), S("in2")}, {S("out1")}),
            LinearLayout({{S("in1"), {{1}, {0}, {2}}}, {S("in2"), {{0}}}},
                         {S("out1")}));
  EXPECT_EQ(l1.sublayout({S("in2"), S("in1")}, {S("out1")}),
            LinearLayout({{S("in1"), {{1}, {0}, {2}}}, {S("in2"), {{0}}}},
                         {S("out1")}));
  EXPECT_EQ(l1.sublayout({S("in2"), S("in1")}, {S("out2"), S("out1")}), l1);
  EXPECT_EQ(l1.sublayout({S("in1")}, {S("out1")}),
            LinearLayout({{S("in1"), {{1}, {0}, {2}}}}, {S("out1")}));
  EXPECT_EQ(l1.sublayout({}, {}), LinearLayout::empty());
  // EXPECT_EQ(l1.sublayout({S("in1")}, {}),
  //           LinearLayout({{S("in1"), {{}, {}, {}}}}, {}));
}


} // namespace
} // namespace mlir::cliff

int main(int argc, char *argv[]) {
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}