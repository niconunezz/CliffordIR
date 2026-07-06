#!/usr/bin/env bash
# compile_case.sh <mask_a> <mask_b> <mask_c> [output_ptx]

set -euo pipefail

MASK_A=$1
MASK_B=$2
MASK_C=$3
NUM_ELS=$4
LAYOUT=$5
OUT_PTX=${6:-output.ptx}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/../../samples/pga3d/geo_prod.mlir"

# ── 1. Instanciar template → 2. MLIR → LLVM dialect → 3. LLVM IR → 4. PTX ──
python3 - <<EOF \
| "$SCRIPT_DIR/../../../build/bin/cliff-opt" --convert-cliffGPU-to-llvm \
| "$SCRIPT_DIR/../../../mlir-translate" --mlir-to-llvmir \
| "$SCRIPT_DIR/../../../llc_mlir" -march=nvptx64 -mcpu=sm_80 -o "$OUT_PTX"

with open("$TEMPLATE") as f:
    src = f.read()

result = src.replace("MASK_A", str($MASK_A)) \
            .replace("MASK_B", str($MASK_B)) \
            .replace("MASK_C", str($MASK_C)) \
            .replace("NUM_ELS", str($NUM_ELS)) \
            .replace("LAYOUT", """$LAYOUT""")

import sys
sys.stdout.write(result)
EOF

echo "✓ PTX generado: $OUT_PTX"