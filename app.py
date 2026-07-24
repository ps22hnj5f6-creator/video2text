#!/usr/bin/env python3
"""短视频批量转文字 - Gradio Web 应用（兼容 Gradio 6.x）"""

import os
import json
import traceback
import tempfile
from pathlib import Path

# HuggingFace 国内镜像（解决 SSL/被墙问题）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gradio as gr

from transcriber import WhisperTranscriber, TranscriptResult, create_transcriber
from downloader import download_video, download_batch, extract_urls, format_share_text


# ========== 全局状态 ==========

class AppState:
    """应用全局状态（模型缓存、配置等）"""
    transcriber = None
    engine = "whisper"
    model_size = "base"
    language = "zh"
    results: list[TranscriptResult] = []

    def get_transcriber(self):
        if self.transcriber is None:
            self.transcriber = create_transcriber(
                self.engine,
                model_size=self.model_size,
                language=self.language
            )
        return self.transcriber


state = AppState()

TEMP_DIR = tempfile.mkdtemp(prefix="v2t_app_")
os.makedirs(TEMP_DIR, exist_ok=True)


# ========== 核心处理逻辑 ==========

def process_uploaded_files(files, model_size, language, progress=gr.Progress()):
    """处理本地上传的视频文件"""
    try:
        if not files:
            return "<p style='color:red'>请上传至少一个视频文件</p>", "", None

        need_reload = (state.model_size != model_size or state.language != language)
        state.model_size = model_size
        state.language = language
        if need_reload:
            state.transcriber = None
        transcriber = state.get_transcriber()

        # Gradio 6.x: file_count="multiple" type="filepath" 返回路径字符串列表
        print(f"[DEBUG] files type: {type(files)}, value: {files}")

        if isinstance(files, list):
            video_paths = [f if isinstance(f, str) else getattr(f, 'name', str(f)) for f in files]
        elif isinstance(files, str):
            video_paths = [files]
        else:
            video_paths = [getattr(files, 'name', str(files))]

        titles = [Path(p).stem for p in video_paths]

        results = []
        total = len(video_paths)

        for i, (path, title) in enumerate(zip(video_paths, titles)):
            progress((i + 1) / total, desc=f"转写 {i+1}/{total}: {title}")
            r = transcriber.transcribe_file(path, title)
            results.append(r)

        state.results = results
        return format_results_table(results), format_results_text(results), generate_export_json(results)

    except Exception as e:
        traceback.print_exc()
        return f"<p style='color:red'>处理出错: {e}</p>", "", None


def process_video_links(text, model_size, language, cookie_path, progress=gr.Progress()):
    """处理粘贴的视频链接"""
    try:
        if not text or not text.strip():
            return "<p style='color:red'>请粘贴至少一个视频链接</p>", "", None

        urls = format_share_text(text.strip())
        if not urls:
            return "<p style='color:red'>未识别到有效链接，请粘贴完整的视频分享链接</p>", "", None

        need_reload = (state.model_size != model_size or state.language != language)
        state.model_size = model_size
        state.language = language
        if need_reload:
            state.transcriber = None

        progress(0.1, desc=f"下载 {len(urls)} 个视频...")
        download_results = download_batch(urls, output_dir=TEMP_DIR,
                                           cookie_file=cookie_path if cookie_path else None)

        video_paths = []
        titles = []
        for dr in download_results:
            if dr.error:
                state.results.append(TranscriptResult(
                    source=dr.url, title=dr.title or dr.url, text="", error=dr.error
                ))
            else:
                video_paths.append(dr.file_path)
                titles.append(dr.title)

        if not video_paths:
            return format_results_table(state.results), format_results_text(state.results), None

        transcriber = state.get_transcriber()
        total_download = len(urls)
        total_transcribe = len(video_paths)

        for i, (path, title) in enumerate(zip(video_paths, titles)):
            step = total_download + i + 1
            total = total_download + total_transcribe
            progress(step / total, desc=f"转写 {i+1}/{total_transcribe}: {title}")
            r = transcriber.transcribe_file(path, title)
            r.source = download_results[i].url if i < len(download_results) else path
            state.results.append(r)

        return format_results_table(state.results), format_results_text(state.results), generate_export_json(state.results)

    except Exception as e:
        traceback.print_exc()
        return f"<p style='color:red'>处理出错: {e}</p>", "", None


# ========== 结果格式化 ==========

def format_results_table(results: list) -> str:
    if not results:
        return "<p>暂无结果</p>"

    html = """<table style="width:100%; border-collapse:collapse; font-size:14px;">
    <tr style="background:#f0f0f0; font-weight:bold;">
        <th style="padding:8px; border:1px solid #ddd; width:5%;">#</th>
        <th style="padding:8px; border:1px solid #ddd; width:25%;">标题/来源</th>
        <th style="padding:8px; border:1px solid #ddd; width:8%;">时长</th>
        <th style="padding:8px; border:1px solid #ddd; width:8%;">字数</th>
        <th style="padding:8px; border:1px solid #ddd; width:44%;">转写内容</th>
        <th style="padding:8px; border:1px solid #ddd; width:8%;">状态</th>
    </tr>"""

    for i, r in enumerate(results):
        status = "✅" if not r.error else "❌"
        content = r.text[:200] + "..." if len(r.text) > 200 else r.text
        if r.error:
            content = f"<span style='color:red'>{r.error}</span>"
        duration_str = f"{r.duration:.1f}s" if r.duration > 0 else "-"
        word_count = len(r.text) if r.text else 0
        source = r.title or r.source[:50]

        bg = "#fff" if i % 2 == 0 else "#f9f9f9"
        html += f"""
    <tr style="background:{bg};">
        <td style="padding:6px; border:1px solid #ddd; text-align:center;">{i+1}</td>
        <td style="padding:6px; border:1px solid #ddd;">{source}</td>
        <td style="padding:6px; border:1px solid #ddd; text-align:center;">{duration_str}</td>
        <td style="padding:6px; border:1px solid #ddd; text-align:center;">{word_count}</td>
        <td style="padding:6px; border:1px solid #ddd;">{content}</td>
        <td style="padding:6px; border:1px solid #ddd; text-align:center;">{status}</td>
    </tr>"""

    html += "</table>"
    return html


def format_results_text(results: list) -> str:
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results):
        if r.error:
            lines.append(f"[{i+1}] {r.title or r.source} — ❌ {r.error}")
        else:
            lines.append(f"[{i+1}] {r.title or r.source} ({r.duration:.1f}s)")
            lines.append(r.text)
            lines.append("")
    return "\n".join(lines)


def generate_export_json(results: list) -> str:
    if not results:
        return None
    export_data = []
    for r in results:
        export_data.append({
            "source": r.source, "title": r.title, "duration": r.duration,
            "text": r.text, "segments": r.segments, "error": r.error
        })
    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
    export_path = os.path.join(TEMP_DIR, "export_results.json")
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    return export_path


def generate_export_txt(results: list) -> str:
    if not results:
        return None
    txt = format_results_text(results)
    export_path = os.path.join(TEMP_DIR, "export_results.txt")
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return export_path


def generate_export_excel(results: list) -> str:
    if not results:
        return None
    try:
        import openpyxl
    except ImportError:
        return generate_export_json(results)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "转写结果"
    ws.append(["序号", "标题/来源", "时长(秒)", "字数", "转写内容", "状态"])

    for i, r in enumerate(results):
        status = "成功" if not r.error else f"失败: {r.error}"
        ws.append([
            i + 1, r.title or r.source[:50],
            round(r.duration, 1), len(r.text) if r.text else 0,
            r.text or r.error or "", status
        ])

    if any(r.segments for r in results):
        ws2 = wb.create_sheet("分段明细")
        ws2.append(["序号", "标题", "开始(秒)", "结束(秒)", "内容"])
        for i, r in enumerate(results):
            for seg in r.segments:
                ws2.append([
                    i + 1, r.title or r.source[:30],
                    seg.get("start", 0), seg.get("end", 0), seg.get("text", "")
                ])

    export_path = os.path.join(TEMP_DIR, "export_results.xlsx")
    wb.save(export_path)
    return export_path


def export_txt_from_state():
    return generate_export_txt(state.results)

def export_json_from_state():
    return generate_export_json(state.results)

def export_excel_from_state():
    return generate_export_excel(state.results)

def clear_results():
    state.results = []
    return "<p>已清空，请上传视频或粘贴链接</p>", "", None


# ========== Gradio 界面 ==========

def build_app():
    with gr.Blocks(title="短视频批量转文字") as app:

        gr.Markdown(
            "# 🎬 短视频批量转文字\n"
            "上传短视频或粘贴分享链接，自动提取语音转写为文字。支持抖音、视频号等平台。\n"
            "视频时长建议 ≤ 90 秒，更长视频也能处理但耗时增加。"
        )

        # 配置区
        with gr.Row():
            model_size = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v3"],
                value="base",
                label="模型大小",
                info="tiny最快但最差，base性价比最佳（推荐），large-v3最准但最慢"
            )
            language = gr.Dropdown(
                choices=["zh", "en", "ja", "ko", "auto"],
                value="zh",
                label="语言",
                info="zh=中文, en=英文, auto=自动检测（稍慢）"
            )

        # 本地上传 Tab
        with gr.Tab("📁 本地上传"):
            file_input = gr.File(
                label="上传短视频",
                file_count="multiple",
                file_types=[".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".3gp"],
                type="filepath"
            )
            upload_btn = gr.Button("🚀 开始转写", variant="primary", size="lg")

        # 链接下载 Tab
        with gr.Tab("🔗 链接下载"):
            gr.Markdown(
                "粘贴视频分享文本或链接（每行一个），支持抖音短链接、视频号链接等。\n\n"
                "抖音示例：`7.47 Dhi:/ 复制打开抖音... https://v.douyin.com/iRNBd6s/`\n\n"
                "⚠️ 抖音/视频号下载可能需要 cookies，详见下方说明。"
            )
            link_input = gr.Textbox(
                label="粘贴分享文本/链接",
                placeholder="每行一个链接，或直接粘贴完整的分享文本...",
                lines=5,
            )
            link_btn = gr.Button("🚀 下载并转写", variant="primary", size="lg")

            with gr.Accordion("Cookies 配置（部分平台需要）", open=False):
                gr.Markdown(
                    "抖音等平台需要登录态才能下载视频。获取方法：\n"
                    "1. 用浏览器登录抖音\n"
                    "2. 安装浏览器插件 'Get cookies.txt LOCALLY'\n"
                    "3. 导出 cookies 为 Netscape 格式文件\n"
                    "4. 将文件路径填入下方"
                )
                cookie_file = gr.Textbox(
                    label="Cookies 文件路径",
                    placeholder="/path/to/cookies.txt",
                    value=""
                )

        # 结果区
        gr.Markdown("## 📋 转写结果")

        result_table = gr.HTML(
            value="<p>暂无结果，请上传视频或粘贴链接</p>",
        )

        result_text = gr.Textbox(
            label="纯文本结果（可复制）",
            value="",
            lines=10,
        )

        # 导出区
        with gr.Row():
            export_txt_btn = gr.Button("📄 导出 TXT")
            export_json_btn = gr.Button("📦 导出 JSON")
            export_excel_btn = gr.Button("📊 导出 Excel")
            clear_btn = gr.Button("🗑️ 清空结果", variant="secondary")

        export_file = gr.File(label="导出文件")

        # 事件绑定
        upload_btn.click(
            fn=process_uploaded_files,
            inputs=[file_input, model_size, language],
            outputs=[result_table, result_text, export_file]
        )

        link_btn.click(
            fn=process_video_links,
            inputs=[link_input, model_size, language, cookie_file],
            outputs=[result_table, result_text, export_file]
        )

        export_txt_btn.click(fn=export_txt_from_state, outputs=export_file)
        export_json_btn.click(fn=export_json_from_state, outputs=export_file)
        export_excel_btn.click(fn=export_excel_from_state, outputs=export_file)

        clear_btn.click(
            fn=clear_results,
            outputs=[result_table, result_text, export_file]
        )

        # 使用说明
        with gr.Accordion("使用说明 & FAQ", open=False):
            gr.Markdown(
                "### 支持的平台\n"
                "- 抖音（需 cookies）\n"
                "- 微信视频号（需 cookies）\n"
                "- B站、YouTube、微博等 yt-dlp 支持的所有平台\n\n"
                "### 性能参考\n"
                "| 模型 | 内存占用 | 90s视频转写耗时(CPU) | 中文准确率 |\n"
                "|------|---------|---------------------|----------|\n"
                "| tiny | ~1GB | ~15s | ~70% |\n"
                "| base | ~1GB | ~30s | ~85% |\n"
                "| small | ~2GB | ~60s | ~92% |\n"
                "| large-v3 | ~3GB | ~180s | ~97% |\n\n"
                "### 注意事项\n"
                "- 首次运行会自动下载模型文件（base约74MB），后续不再重复下载\n"
                "- CPU模式足够应对90s短视频，GPU可加速4-10倍\n"
                "- 证券金融术语建议用 base 或以上模型，tiny容易出错\n"
                "- 如需更高准确率，可后续切换到腾讯云ASR引擎"
            )

    return app


# ========== 启动 ==========

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=".contain { max-width: 1200px; margin: auto; } .result-box { min-height: 200px; }"
    )
