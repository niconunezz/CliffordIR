#!/usr/bin/env bash
# compile_case.sh <mask_a> <mask_b> <mask_c> [output_ptx]

set -euo pipefail

NUM_ELS=$1
LAYOUT=$2
OUT_PTX=${3:-output.ptx}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/../samples/cliff_to_llvm.mlir"

TMP_MLIR=$(mktemp /tmp/cliff_case_XXXXXX.mlir)
echo "[tmp] TMP_MLIR=$TMP_MLIR"

TMP_INTER_MLIR=$(mktemp /tmp/cliff_inter_XXXXXX.mlir)

TMP_LLVM_MLIR=$(mktemp /tmp/cliff_llvm_XXXXXX.mlir)
echo "[tmp] TMP_LLVM_MLIR=$TMP_LLVM_MLIR"

TMP_LL=$(mktemp /tmp/cliff_XXXXXX.ll)
echo "[tmp] TMP_LL=$TMP_LL"

cleanup() { rm -f "$TMP_MLIR" "$TMP_INTER_MLIR" "$TMP_LLVM_MLIR" "$TMP_LL"; }
trap cleanup EXIT

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
    --rewrite-sandwich --rewrite-exponential --geometric-type-conversion --convert-cliff-to-cliffGPU > "$TMP_INTER_MLIR"

python3 - <<EOF
import re, sys

with open("$TMP_INTER_MLIR") as f:
    src = f.read()

src = re.sub(
    r'(#clg\.linear<)\{.*?\}(>)',
    rf'\1{"""$LAYOUT"""}\2',
    src
)

with open("$TMP_INTER_MLIR", "w") as f:
    f.write(src)
EOF

"$SCRIPT_DIR/../../build/bin/cliff-opt" "$TMP_INTER_MLIR" \
     --convert-cliffGPU-to-llvm > "$TMP_LLVM_MLIR"

# ── 3. LLVM dialect → LLVM IR ────────────────────────────────────
"$SCRIPT_DIR/../../mlir-translate" --mlir-to-llvmir "$TMP_LLVM_MLIR" -o "$TMP_LL"

# ── 4. LLVM IR → PTX ─────────────────────────────────────────────
"$SCRIPT_DIR/../../llc_mlir" -march=nvptx64 -mcpu=sm_80 "$TMP_LL" -o "$OUT_PTX"

echo "✓ PTX generated: $OUT_PTX"