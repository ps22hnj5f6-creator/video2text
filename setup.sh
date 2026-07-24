#!/bin/bash
# 短视频批量转文字 - 一键安装脚本
# 适用于 macOS (Apple Silicon / Intel) / Linux

set -e

echo "========================================"
echo "  短视频批量转文字 - 安装脚本"
echo "========================================"

# 1. 检查 Python
echo ""
echo "[1/4] 检查 Python 环境..."
PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON=$(which python3)
    echo "  ✅ 找到 Python: $PYTHON ($(python3 --version))"
elif command -v python &>/dev/null; then
    PYTHON=$(which python)
    echo "  ✅ 找到 Python: $PYTHON ($(python --version))"
else
    echo "  ❌ 未找到 Python，请先安装 Python 3.9+"
    echo "     macOS: brew install python"
    exit 1
fi

VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
MAJOR=$(echo $VERSION | cut -d. -f1)
MINOR=$(echo $VERSION | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || [ "$MINOR" -lt 9 ]; then
    echo "  ❌ Python版本过低 ($VERSION)，需要 3.9+"
    exit 1
fi

# 2. 创建虚拟环境
echo ""
echo "[2/4] 创建虚拟环境..."
VENV_DIR="$(cd "$(dirname "$0")" && pwd)/.venv"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    echo "  ✅ 虚拟环境创建完成: $VENV_DIR"
else
    echo "  ⏭️ 虚拟环境已存在，跳过创建"
fi

# 激活
source "$VENV_DIR/bin/activate"

# 3. 安装依赖
echo ""
echo "[3/4] 安装 Python 依赖（可能需要5-10分钟）..."
pip install --upgrade pip -q
pip install -r "$(cd "$(dirname "$0")" && pwd)/requirements.txt"

echo "  ✅ Python 依赖安装完成"

# 4. 检查 ffmpeg
echo ""
echo "[4/4] 检查 ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    echo "  ✅ 系统 ffmpeg 可用"
elif python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" &>/dev/null; then
    echo "  ✅ imageio-ffmpeg 自带 ffmpeg 可用"
else
    echo "  ⚠️ ffmpeg 未找到"
    echo "     imageio-ffmpeg 将在首次运行时自动提供 ffmpeg"
    echo "     如需系统级 ffmpeg: macOS→brew install ffmpeg"
fi

# 完成
echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "启动应用："
echo "  cd $(cd "$(dirname "$0")" && pwd)"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo ""
echo "应用将在 http://localhost:7860 打开"
