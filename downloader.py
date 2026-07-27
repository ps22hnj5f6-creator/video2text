#!/usr/bin/env python3
"""视频下载模块 - 支持抖音、视频号等平台"""

import os
import re
import sys
import tempfile
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class DownloadResult:
    """下载结果"""
    url: str                  # 原始链接
    file_path: str            # 下载后的本地文件路径
    title: str                # 视频标题
    duration: float = 0.0     # 视频时长
    error: Optional[str] = None


# 抖音链接正则（匹配短链接和分享文本中的链接）
DOUYIN_URL_PATTERN = re.compile(
    r'(https?://v\.douyin\.com/[A-Za-z0-9_\-]+|https?://www\.douyin\.com/video/\d+)'
)

# 视频号链接正则
CHANNEL_URL_PATTERN = re.compile(
    r'(https?://channels.weixin.qq.com/[^\s<>"{}|\\^`\[\]]+)'
)

# 通用URL正则
GENERAL_URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+'
)

# 常见需要 cookies 的平台
COOKIE_REQUIRED_PLATFORMS = ["douyin", "douyinvod", "iesdouyin",
                             "channels.weixin.qq.com", "weixin.qq.com"]


def _get_yt_dlp_cmd() -> list[str]:
    """返回可调用的 yt-dlp 命令列表

    优先使用当前 Python 解释器以模块方式调用，避免 venv/Docker 中 PATH 问题。
    """
    return [sys.executable, "-m", "yt_dlp"]


def _yt_dlp_available() -> bool:
    """检查 yt-dlp 是否可用"""
    try:
        proc = subprocess.run(
            _get_yt_dlp_cmd() + ["--version"],
            capture_output=True, text=True, timeout=10
        )
        return proc.returncode == 0
    except Exception:
        return False


def extract_urls(text: str) -> list[str]:
    """从文本中提取所有视频链接"""
    # 先找抖音链接
    douyin_urls = DOUYIN_URL_PATTERN.findall(text)
    # 再找视频号链接
    channel_urls = CHANNEL_URL_PATTERN.findall(text)
    # 最后找其他链接
    other_urls = GENERAL_URL_PATTERN.findall(text)

    seen = set()
    all_urls = []
    for u in douyin_urls + channel_urls + other_urls:
        # 去掉末尾的标点、斜杠等干扰字符
        clean = u.rstrip("/.,;:!?）}")
        if clean not in seen:
            seen.add(clean)
            all_urls.append(clean)
    return all_urls


def _clean_yt_dlp_error(stderr: str, url: str) -> str:
    """把 yt-dlp 的 stderr 整理成用户友好的错误信息"""
    if not stderr:
        return "下载失败，未知错误"

    stderr_lower = stderr.lower()

    # cookies 相关错误
    if any(k in stderr_lower for k in ["cookies", "cookie"]):
        if "douyin" in url.lower() or "v.douyin.com" in url.lower():
            return "抖音下载需要 cookies：请配置抖音登录后的 cookies.txt 文件（详见页面下方说明）"
        if "channels.weixin" in url.lower():
            return "视频号下载需要 cookies：请配置微信登录后的 cookies.txt 文件"
        return "该平台下载需要 cookies，请在下方配置 cookies.txt 文件后重试"

    # 版权/地区/私有限制
    if any(k in stderr_lower for k in ["private", "unavailable", "restricted", "removed"]):
        return "视频不可访问（可能为私密、已删除或地区受限）"

    # 登录相关
    if any(k in stderr_lower for k in ["login", "sign in", "authentication"]):
        return "下载需要登录态，请在下方配置 cookies.txt 文件后重试"

    # 网络/超时
    if any(k in stderr_lower for k in ["timeout", "timed out", "connection", "unreachable"]):
        return "网络连接超时，请检查链接是否有效或稍后重试"

    # 取第一行 ERROR
    for line in stderr.splitlines():
        if "ERROR" in line:
            return line.split(":", 2)[-1].strip()[:300]

    return stderr.strip()[:300]


def _needs_cookies(url: str) -> bool:
    """判断给定 URL 是否大概率需要 cookies"""
    url_lower = url.lower()
    return any(p in url_lower for p in COOKIE_REQUIRED_PLATFORMS)


def download_video(url: str, output_dir: str = None,
                   cookie_file: str = None) -> DownloadResult:
    """用 yt-dlp 下载视频

    Args:
        url: 视频链接
        output_dir: 输出目录（默认临时目录）
        cookie_file: cookies文件路径（某些平台需要登录态）
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="v2t_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    result = DownloadResult(url=url, file_path="", title="")

    if not _yt_dlp_available():
        result.error = "yt-dlp 未安装！请运行: pip install yt-dlp"
        return result

    # 对需要 cookies 的平台提前提示
    if not cookie_file and _needs_cookies(url):
        platform = "抖音" if "douyin" in url.lower() else "视频号"
        result.error = f"{platform}下载需要 cookies：请在下方配置 cookies.txt 文件后重试"
        return result

    cmd = _get_yt_dlp_cmd() + [
        "--no-check-certificates",
        "--no-warnings",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--referer", url,
        "-f", "best[ext=mp4]/best",       # 优先mp4格式
        "--max-filesize", "100M",          # 90s短视频不会超过100M
        "-o", os.path.join(output_dir, "%(title)s.%(ext)s"),
        "--print", "after_move:filepath",  # 输出最终文件路径
        "--print", "title",                # 输出标题
        url
    ]

    if cookie_file:
        cmd.extend(["--cookies", cookie_file])

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        # 收集 stdout
        output_lines = proc.stdout.strip().split("\n") if proc.stdout.strip() else []

        if proc.returncode != 0:
            result.error = _clean_yt_dlp_error(proc.stderr, url)
            return result

        # 解析输出：yt-dlp 按 --print 顺序输出
        # 第一行是 title，第二行是 filepath
        if len(output_lines) >= 2:
            result.title = output_lines[0].strip()
            result.file_path = output_lines[1].strip()
        elif len(output_lines) == 1:
            # 只有一行输出，尝试在目录中找文件
            result.file_path = output_lines[0].strip()
            result.title = Path(result.file_path).stem if result.file_path else url

        # 如果 yt-dlp 没输出有效路径，扫描输出目录
        if not result.file_path or not os.path.exists(result.file_path):
            files = sorted(Path(output_dir).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            video_files = [f for f in files if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov", ".flv", ".3gp")]
            if video_files:
                result.file_path = str(video_files[0])
                result.title = video_files[0].stem

        if not result.file_path:
            result.error = "下载完成但未找到视频文件"
        elif not os.path.exists(result.file_path):
            result.error = f"下载文件不存在: {result.file_path}"

    except subprocess.TimeoutExpired:
        result.error = "下载超时（120s），短视频链接可能无效"
    except FileNotFoundError:
        result.error = "yt-dlp 未安装！请运行: pip install yt-dlp"
    except Exception as e:
        result.error = str(e)

    return result


def download_batch(urls: list[str], output_dir: str = None,
                   cookie_file: str = None) -> list[DownloadResult]:
    """批量下载多个视频链接"""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="v2t_batch_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] 下载: {url}")
        r = download_video(url, output_dir, cookie_file)
        results.append(r)
        if r.error:
            print(f"  ❌ 失败: {r.error}")
        else:
            print(f"  ✅ 完成: {r.title}")

    return results


def format_share_text(text: str) -> list[str]:
    """解析用户粘贴的分享文本，提取视频链接

    抖音分享文本格式示例：
    "7.47 Dhi:/ 复制打开抖音，看看【证券分析师小王的作品】... https://v.douyin.com/iRNBd6s/"

    视频号分享文本格式示例：
    "视频号: xxx https://..."
    """
    return extract_urls(text)


if __name__ == "__main__":
    # 简单自测
    test_url = "https://v.douyin.com/vndqkfMiG60/"
    print("测试下载:", test_url)
    r = download_video(test_url)
    print(r)
