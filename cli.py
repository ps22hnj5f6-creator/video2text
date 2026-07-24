#!/usr/bin/env python3
"""命令行快速测试 - 不依赖Gradio，验证核心转写逻辑"""

import sys
import argparse
from pathlib import Path

from transcriber import create_transcriber
from downloader import download_video, extract_urls


def main():
    parser = argparse.ArgumentParser(description="短视频转文字 - 命令行版")
    parser.add_argument("input", nargs="+", help="视频文件路径或链接")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--lang", default="zh", choices=["zh", "en", "ja", "ko", "auto"])
    parser.add_argument("--cookies", default=None, help="cookies文件路径（抖音/视频号需要)")
    parser.add_argument("--output", default=None, help="输出文件路径（默认stdout)")
    args = parser.parse_args()

    transcriber = create_transcriber("whisper", model_size=args.model, language=args.lang)

    # 分类：本地文件 vs 链接
    local_files = []
    urls = []
    for inp in args.input:
        if Path(inp).exists():
            local_files.append(inp)
        elif inp.startswith("http"):
            urls.append(inp)
        else:
            # 可能是含链接的文本
            found_urls = extract_urls(inp)
            if found_urls:
                urls.extend(found_urls)
            else:
                print(f"⚠️ 无法识别: {inp}")

    # 下载链接视频
    downloaded = []
    for url in urls:
        print(f"下载: {url}")
        r = download_video(url, cookie_file=args.cookies)
        if r.error:
            print(f"  ❌ 下载失败: {r.error}")
        else:
            print(f"  ✅ 已下载: {r.title}")
            downloaded.append(r.file_path)

    # 合并所有本地文件路径
    all_files = local_files + downloaded
    titles = [Path(f).stem for f in all_files]

    if not all_files:
        print("❌ 无可处理的视频文件")
        return

    # 批量转写
    results = transcriber.transcribe_batch(all_files, titles)

    # 输出
    output_lines = []
    for i, r in enumerate(results):
        if r.error:
            output_lines.append(f"[{i+1}] ❌ {r.title}: {r.error}")
        else:
            output_lines.append(f"[{i+1}] {r.title} ({r.duration:.1f}s)")
            output_lines.append(r.text)
            output_lines.append("")

    output_text = "\n".join(output_lines)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\n结果已保存到: {args.output}")
    else:
        print("\n" + output_text)


if __name__ == "__main__":
    main()
