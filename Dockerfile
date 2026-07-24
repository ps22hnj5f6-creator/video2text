FROM python:3.11-slim

WORKDIR /app

# === 国内加速源 ===
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# === 系统依赖 ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# === Python 依赖 ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === 构建时预下载 faster-whisper base 模型 ===
# 使用国内 HF 镜像，构建时直接下载到镜像内
# 这样容器启动时不需要再下载，秒开
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/app/hf_cache
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" \
    && echo "模型预下载完成"

# === 应用代码 ===
COPY . .

# === 健康检查（Sealos 自动检测容器状态）===
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

# Gradio 6.x 默认不开 inbrowser（容器环境不需要）
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
