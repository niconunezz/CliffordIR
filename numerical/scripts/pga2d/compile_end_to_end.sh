#!/usr/bin/env bash
set -euo pipefail

NUM_ELS=$1
LAYOUT=$2
OUT_PTX=${3:-output.ptx}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/../../samples/pga2d/cliff_to_llvm.mlir"
BUILD="$SCRIPT_DIR/../../../build/bin"

TMP_INTER=$(mktemp /tmp/cliff_inter_XXXXXX.mlir)
cleanup() { rm -f "$TMP_INTER"; }
trap cleanup EXIT

t() { date +%s%N; }
elapsed() { echo "$(( ($2 - $1) / 1000000 ))ms"; }

t0=$(t)
sed "s/NUM_ELS/$NUM_ELS/g" "$TEMPLATE" \
| "$BUILD/cliff-opt" --mlir-timing \
    --rewrite-sandwich --rewrite-exponential \
    --geometric-type-conversion --convert-cliff-to-cliffGPU - \
| python3 -c "
import sys, re
src = sys.stdin.read()
print(re.sub(r'(#clg\.linear<)\{.*?\}(>)', r'\1${LAYOUT}\2', src))
" > "$TMP_INTER"
t1=$(t)
echo "── pass1+layout: $(elapsed $t0 $t1)"

t2=$(t)
"$BUILD/cliff-opt" --mlir-timing \
    --convert-cliffGPU-to-llvm "$TMP_INTER" \
| "$SCRIPT_DIR/../../../mlir-translate" --mlir-to-llvmir - \
| "$SCRIPT_DIR/../../../llc_mlir" -march=nvptx64 -mcpu=sm_80 -O1 - -o "$OUT_PTX"
t3=$(t)
echo "── pass2+translate+llc: $(elapsed $t2 $t3)"
echo "── total: $(elapsed $t0 $t3)"