# ---- 基础镜像：CUDA + Ubuntu 22.04 ----
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# ----- 环境变量 -----
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_OFFLINE=1

# ---- 安装依赖 ----
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 复制代码/模型 ----
COPY model-easy/   ./model-easy/
COPY model-medium/ ./model-medium/
COPY model-hard/   ./model-hard/
COPY predict.py    ./predict.py
