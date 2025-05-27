#!/usr/bin/env bash
set -e

INPUT_FILE="$1"
OUTPUT_DIR="$2"


echo "✓ inputfile: $INPUT_FILE"
echo "✓ ouputfile: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# 🔁 显式使用绝对路径，避免找不到文件
python -u /workspace/predict.py \
    # --dataset "$INPUT_FILE" \
    # --output "$OUTPUT_DIR"
