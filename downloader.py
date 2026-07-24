#!/usr/bin/env python3
"""视频下载模块 - 支持抖音、视频号等平台"""

import os
import re
import tempfile
import subprocess
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
    r'(https?://v\.douyin\.com/[A-Za-z0-9]+|https?://www\.douyin\.com/video/\d+)'
)

# 通用URL正则
GENERAL_URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+'
)


def extract_urls(text: str) -> list[str]:
    """从文本中提取所有视频链接"""
    # 先找抖音链接
    douyin_urls = DOUYIN_URL_PATTERN.findall(text)
    # 再找其他链接
    other_urls = GENERAL_URL_PATTERN.findall(text)

    # 合并去重（抖音链接优先）
    all_urls = douyin_urls + [u for u in other_urls if u not in douyin_urls]
    return all_urls


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

    result = DownloadResult(url=url, file_path="", title="")

    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "-f", "best[ext=mp4]/best",       # 优先mp4格式
        "--max-filesize", "100M",          # 90s短视频不会超过100M
        "-o", os.path.join(output_dir, "%(title)s.%(ext)s"),
        "--print", "after_move:filepath",  # 输出最终文件路径
        "--print", "title",                # 输出标题
        "--no-warnings",
        url
    ]

    # 抖音/视频号需要cookies
    if cookie_file:
        cmd.extend(["--cookies", cookie_file])
    # 抖音特殊处理
    if "douyin" in url:
        cmd.extend([
            "--extractor", "douyin",
            "--extractor-args", "douyin:api_hostname=api16-normal-v4.douyinvod.com"
        ])

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if proc.returncode != 0:
            # yt-dlp 经常通过 stderr 输出正常信息，需区分真错误
            error_lines = [l for l in proc.stderr.split("\n")
                          if "ERROR" in l and "WARNING" not in l]
            if error_lines:
                result.error = f"下载失败: {error_lines[0][:300]}"
                return result

        # 解析输出
        output_lines = proc.stdout.strip().split("\n")
        if len(output_lines) >= 2:
            result.title = output_lines[0].strip()
            result.file_path = output_lines[1].strip()
        elif len(output_lines) == 1:
            # 只有一行输出，尝试在目录中找文件
            result.file_path = output_lines[0].strip()
            result.title = Path(result.file_path).stem if result.file_path else url

        # 如果 yt-dlp 没输出路径，扫描输出目录
        if not result.file_path or not os.path.exists(result.file_path):
            files = list(Path(output_dir).glob("*.mp4"))
            if files:
                result.file_path = str(files[0])
                result.title = files[0].stem
            else:
                # 尝试其他视频格式
                for ext in ["*.webm", "*.mkv", "*.mov", "*.flv"]:
                    files = list(Path(output_dir).glob(ext))
                    if files:
                        result.file_path = str(files[0])
                        result.title = files[0].stem
                        break

        if not result.file_path:
            result.error = "下载完成但未找到视频文件"

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
