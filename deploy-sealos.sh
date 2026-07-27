#!/usr/bin/env bash
# === video2text 一键部署到 Sealos（使用 GHCR 镜像）===
set -e

# =============================================
# ⚙️ 配置区 —— 改这里就行
# =============================================
GHCR_USER="ps22hnj5f6-creator"      # GitHub 用户名 / 组织名
IMAGE_NAME="video2text"             # 镜像名
IMAGE_TAG="latest"                  # 镜像标签
FULL_IMAGE="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

# 应用资源（根据实际流量调整）
CPU_REQUEST="1"
CPU_LIMIT="2"
MEMORY_REQUEST="2Gi"
MEMORY_LIMIT="4Gi"

# Sealos 上的应用名
APP_NAME="video2text"

# =============================================
# DeepSeek API Key（通过环境变量注入，不写死在仓库里）
# 用法二选一：
#   1) 运行本脚本前先 export DEEPSEEK_API_KEY=sk-xxx，生成的 YAML 会自动带上；
#   2) 或在 Sealos 应用的环境变量里直接配置 DEEPSEEK_API_KEY（推荐，持久且安全）。
# 注意：此前聊天中贴出的 Key 已暴露，建议去 https://platform.deepseek.com 重置。
# =============================================
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

echo "========================================="
echo "  video2text Sealos 部署脚本"
echo "========================================="
echo "镜像地址: ${FULL_IMAGE}"
echo ""

# =============================================
# Step 1: 确认 sealos CLI 已登录
# =============================================
if ! command -v sealos &> /dev/null; then
    echo "⚠️ 未检测到 sealos CLI，下面提供手动部署 YAML。"
    echo "   如需自动部署，请先安装 sealos CLI 并登录："
    echo "   https://sealos.io/docs/self-hosting/lifecycle-management/"
fi

# =============================================
# Step 2: 输出 Deployment YAML
# =============================================
echo ""
echo "========================================="
echo "  🎉 Kubernetes 部署 YAML"
echo "========================================="
echo ""
echo "复制以下内容到 Sealos 终端（kubectl apply -f）即可部署/更新："
echo ""

cat <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}
  labels:
    app: ${APP_NAME}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${APP_NAME}
  template:
    metadata:
      labels:
        app: ${APP_NAME}
    spec:
      containers:
      - name: ${APP_NAME}
        image: ${FULL_IMAGE}
        imagePullPolicy: Always
        ports:
        - containerPort: 7860
        resources:
          requests:
            cpu: "${CPU_REQUEST}"
            memory: "${MEMORY_REQUEST}"
          limits:
            cpu: "${CPU_LIMIT}"
            memory: "${MEMORY_LIMIT}"
        env:
        - name: HF_ENDPOINT
          value: "https://hf-mirror.com"
        - name: DEEPSEEK_API_KEY
          value: "${DEEPSEEK_API_KEY}"
        - name: GRADIO_SERVER_NAME
          value: "0.0.0.0"
        - name: GRADIO_SERVER_PORT
          value: "7860"
        livenessProbe:
          httpGet:
            path: /
            port: 7860
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /
            port: 7860
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ${APP_NAME}-service
spec:
  selector:
    app: ${APP_NAME}
  ports:
  - port: 7860
    targetPort: 7860
EOF

echo ""
echo "========================================="
echo "  Sealos Web UI 快速部署步骤"
echo "========================================="
echo ""
echo "1. 登录 Sealos 公有云: https://cloud.sealos.io"
echo "2. 进入「应用管理」> 找到已部署的 ${APP_NAME}"
echo "3. 点击「重新部署」或「更新」"
echo "4. 镜像地址填写: ${FULL_IMAGE}"
echo "5. 容器端口: 7860"
echo "6. 环境变量确认包含: HF_ENDPOINT=https://hf-mirror.com"
echo "7. 点击「部署/更新」，等待 30-60 秒容器启动"
echo "8. 访问 Sealos 分配的公网域名即可使用"
echo ""
echo "========================================="
echo "  完成！有问题随时问我"
echo "========================================="
