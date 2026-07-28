FROM python:3.11-slim

WORKDIR /app

# === 国内加速源 ===
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# === 系统依赖 ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version | head -1

# === Python 依赖 ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === 构建时预下载 faster-whisper base 模型 ===
# 构建环境（GitHub Actions）在海外，直接用 HuggingFace 官方源
# 运行环境（Sealos/国内）需要国内镜像，通过运行时环境变量 HF_ENDPOINT 设置
ENV HF_HOME=/app/hf_cache
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" \
    && echo "模型预下载完成"

# === 应用代码 ===
COPY . .

# === 健康检查（Sealos 自动检测容器状态）===
# 使用独立的 /health 端点，不经过 Gradio Blocks，避免大文件上传阻塞主线程时
# 健康检查拿不到响应。同时延长 start-period 和 interval，给模型加载留足时间。
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

# Gradio 6.x 默认不开 inbrowser（容器环境不需要）
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
