#!/usr/bin/env bash
# === video2text 一键构建 + 推送阿里云 ACR + Sealos 部署指南 ===
set -e

# =============================================
# ⚙️ 配置区 —— 改这里就行
# =============================================
ACR_REGION="cn-hangzhou"            # 华东1（杭州）
ACR_NAMESPACE="video2text"          # ACR 命名空间（需提前在 ACR 控制台创建）
IMAGE_NAME="video2text"             # 镜像名
IMAGE_TAG="latest"                  # 镜像标签
ACR_REGISTRY="${ACR_REGION}.aliyuncs.com"  # ACR 仓库地址

# 完整镜像地址（Sealos 部署时用这个）
FULL_IMAGE="${ACR_REGISTRY}/${ACR_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "========================================="
echo "  video2text 构建 & 推送"
echo "========================================="
echo "镜像地址: ${FULL_IMAGE}"
echo ""

# =============================================
# Step 1: 构建 Docker 镜像
# =============================================
echo "[1/3] 构建 Docker 镜像..."
docker build -t "${FULL_IMAGE}" .

echo "✅ 构建完成"

# =============================================
# Step 2: 登录阿里云 ACR & 推送
# =============================================
echo "[2/3] 推送到阿里云 ACR..."
echo "请输入阿里云 ACR 登录密码（在 ACR 控制台 > 访问凭证 > 设置固定密码）："
docker login "${ACR_REGISTRY}"

docker push "${FULL_IMAGE}"
echo "✅ 推送完成"

# =============================================
# Step 3: Sealos 部署指南
# =============================================
echo ""
echo "========================================="
echo "  🎉 部署到 Sealos"
echo "========================================="
echo ""
echo "镜像已推送到: ${FULL_IMAGE}"
echo ""
echo "=== Sealos 部署步骤 ==="
echo ""
echo "方式 A —— Sealos Web UI（最简单）："
echo "  1. 登录 Sealos 公有云: https://cloud.sealos.io"
echo "  2. 进入「应用管理」>「应用部署」"
echo "  3. 填写："
echo "     - 镜像地址: ${FULL_IMAGE}"
echo "     - 容器端口: 7860"
echo "     - CPU: 2核（最低1核，推荐2核保证转写速度）"
echo "     - 内存: 4GB（模型+转写约1.5GB，建议4GB留余量）"
echo "     - 网络: 开启「公网访问」，Sealos自动分配域名"
echo "  4. 点「部署」，等30秒容器启动"
echo "  5. 访问 Sealos 分配的公网域名即可使用"
echo ""
echo "方式 B —— Sealos 终端 (kubectl)："
echo "  1. 登录 Sealos，打开「终端」"
echo "  2. 粘贴以下 YAML："
echo ""
cat <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: video2text
  labels:
    app: video2text
spec:
  replicas: 1
  selector:
    matchLabels:
      app: video2text
  template:
    metadata:
      labels:
        app: video2text
    spec:
      containers:
      - name: video2text
        image: ${FULL_IMAGE}
        ports:
        - containerPort: 7860
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: HF_ENDPOINT
          value: "https://hf-mirror.com"
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
  name: video2text-service
spec:
  selector:
    app: video2text
  ports:
  - port: 7860
    targetPort: 7860
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: video2text-ingress
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: video2text-service
            port:
              number: 7860
EOF

echo ""
echo "=== ACR 配置提醒 ==="
echo "  如果还没创建 ACR 命名空间："
echo "  登录 https://cr.console.aliyun.com"
echo "  > 命名空间 > 创建命名空间: ${ACR_NAMESPACE}"
echo ""
echo "========================================="
echo "  完成！有问题随时问我"
echo "========================================="
