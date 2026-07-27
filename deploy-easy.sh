#!/usr/bin/env bash
# === video2text Sealos 部署 —— 傻瓜版 ===
# 只需要 3 步：构建 → 推送 → 部署

set -e

# =============================================
# 改这两行就行！
# =============================================
ACR_LOGIN_REGISTRY="registry.cn-hangzhou.aliyuncs.com"   # ACR 登录地址（标准入口）
ACR_PUSH_REGISTRY="crpi-48rfadbeb000k7yc8.cn-hangzhou.personal.cr.aliyuncs.com"   # ACR 个人版推送地址
ACR_NAMESPACE="video2text"                # ACR 命名空间（需提前在阿里云创建）

IMAGE="${ACR_PUSH_REGISTRY}/${ACR_NAMESPACE}/video2text:latest"

echo "=== Step 1: 构建镜像 ==="
cd "$(dirname "$0")"
docker build -t "$IMAGE" .
echo "✅ 构建完成"

echo ""
echo "=== Step 2: 推送到阿里云 ACR ==="
echo "需要输入 ACR 密码（在 https://cr.console.aliyun.com > 访问凭证 获取）"
docker login "$ACR_LOGIN_REGISTRY"
docker push "$IMAGE"
echo "✅ 推送完成"

echo ""
echo "=== Step 3: 在 Sealos 部署 ==="
echo "1. 打开 https://cloud.sealos.io"
echo "2. 点「应用管理」>「应用部署」"
echo "3. 填写："
echo "   镜像: $IMAGE"
echo "   端口: 7860"
echo "   CPU: 2核 | 内存: 4GB"
echo "   开启公网访问"
echo "4. 点「部署」→ 等30秒 → 搞定"
echo ""
echo "🎉 部署地址就是 Sealos 给你的公网域名"
