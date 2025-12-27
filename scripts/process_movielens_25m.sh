#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

# Python 解释器
PYTHON="/root/miniconda3/envs/pytorch291/bin/python"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 设置 Python 模块搜索路径
export PYTHONPATH="$PROJECT_ROOT"

# 运行脚本
"$PYTHON" src/scripts/preprocess_raw_data.py \
  --config "./config/movieLens_25m.yaml"
