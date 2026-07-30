#!/usr/bin/env bash
set -euo pipefail

MASK_A=$1
MASK_B=$2
MASK_C=$3
MASK_D=$4
NUM_ELS=$5
LAYOUT=$6
OUT_PTX=${7:-output.ptx}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/../../samples/pga3d/cliff_to_llvm.mlir"
BUILD="$SCRIPT_DIR/../../../build/bin"

TMP_INTER=$(mktemp /tmp/cliff_inter_XXXXXX.mlir)
cleanup() { rm -f "$TMP_INTER"; }
trap cleanup EXIT

sed "s/NUM_ELS/$NUM_ELS/g" "$TEMPLATE" \
| "$BUILD/cliff-opt" \
    --rewrite-sandwich --rewrite-exponential \
    --geometric-type-conversion --convert-cliff-to-cliffGPU - \
| python3 -c "
import sys, re
src = sys.stdin.read()
print(re.sub(r'(#clg\.linear<)\{.*?\}(>)', r'\1${LAYOUT}\2', src))
" > "$TMP_INTER"

"$BUILD/cliff-opt" \
    --convert-cliffGPU-to-llvm "$TMP_INTER" \
| "$SCRIPT_DIR/../../../mlir-translate" --mlir-to-llvmir - \
| "$SCRIPT_DIR/../../../llc_mlir" -march=nvptx64 -mcpu=sm_80 -O1 - -o "$OUT_PTX"
