#!/usr/bin/env bash
# compile_case.sh <mask_a> <mask_b> <mask_c> [output_ptx]

set -euo pipefail

NUM_ELS=$1
OUT_PTX=${2:-output.ptx}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/../samples/cliff_to_llvm.mlir"

TMP_MLIR=$(mktemp /tmp/cliff_case_XXXXXX.mlir)
echo "[tmp] TMP_MLIR=$TMP_MLIR"

TMP_LLVM_MLIR=$(mktemp /tmp/cliff_llvm_XXXXXX.mlir)
echo "[tmp] TMP_LLVM_MLIR=$TMP_LLVM_MLIR"

TMP_LL=$(mktemp /tmp/cliff_XXXXXX.ll)
echo "[tmp] TMP_LL=$TMP_LL"

# cleanup() { rm -f "$TMP_MLIR" "$TMP_LLVM_MLIR" "$TMP_LL"; }
# trap cleanup EXIT

python3 - <<EOF
import re, sys

with open("$TEMPLATE") as f:
    src = f.read()

masks = [$NUM_ELS]
result = src.replace("NUM_ELS", str($NUM_ELS))

with open("$TMP_MLIR", "w") as f:
    f.write(result)
EOF

# ── 2. MLIR → LLVM dialect ───────────────────────────────────────
"$SCRIPT_DIR/../../build/bin/cliff-opt" "$TMP_MLIR" \
    --rewrite-sandwich --rewrite-exponential --geometric-type-conversion --convert-cliff-to-cliffGPU --convert-cliffGPU-to-llvm > "$TMP_LLVM_MLIR"

# ── 3. LLVM dialect → LLVM IR ────────────────────────────────────
"$SCRIPT_DIR/../../mlir-translate" --mlir-to-llvmir "$TMP_LLVM_MLIR" -o "$TMP_LL"

# ── 4. LLVM IR → PTX ─────────────────────────────────────────────
"$SCRIPT_DIR/../../llc_mlir" -march=nvptx64 -mcpu=sm_80 "$TMP_LL" -o "$OUT_PTX"

echo "✓ PTX generated: $OUT_PTX"