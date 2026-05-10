#!/bin/bash

# 1. 定义时间和输出目录
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="./output"

# 2. 确保目录存在 (mkdir -p 如果目录已存在不会报错)
mkdir -p "$OUTPUT_DIR"

# 3. 设置日志文件路径
LOG_FILE="$OUTPUT_DIR/predict_stdout-$TIMESTAMP.log"

echo "日志将保存到: $LOG_FILE"

# 4. 运行训练
CUDA_VISIBLE_DEVICES=1 python ./predict.py |& tee -a "$LOG_FILE"