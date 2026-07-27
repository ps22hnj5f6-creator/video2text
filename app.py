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
        # 每个文件3子步骤：提取音频(1) + 转写(2) + 完成(3)，首次加载模型额外1步
        model_load_step = 1 if state.transcriber is None else 0
        sub_steps_per_file = 3
        total_steps = total * sub_steps_per_file + model_load_step
        current_step = 0

        # 首次加载模型单独显示进度
        if model_load_step:
            current_step += 1
            progress(current_step / total_steps, desc="正在加载 Whisper 模型（首次需下载）...")

        transcriber = state.get_transcriber()

        for i, (path, title) in enumerate(zip(video_paths, titles)):
            r = TranscriptResult(source=path, title=title)

            # 子步骤1：提取音频（快速）
            current_step += 1
            progress(current_step / total_steps, desc=f"[{i+1}/{total}] 提取音频: {title}")

            try:
                audio_path, duration = transcriber.prepare_audio(path)
                r.duration = duration

                # 子步骤2：转写（耗时最长）
                current_step += 1
                progress(current_step / total_steps, desc=f"[{i+1}/{total}] 正在转写: {title}")

                full_text, segment_list = transcriber.transcribe_audio(audio_path)
                r.text = full_text
                r.segments = segment_list

                # 清理临时音频
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

                # 子步骤3：完成
                current_step += 1
                progress(current_step / total_steps, desc=f"[{i+1}/{total}] ✅ 完成: {title}")

            except Exception as e:
                r.error = str(e)
                current_step += 2  # 跳过未完成的步骤
                progress(current_step / total_steps, desc=f"[{i+1}/{total}] ❌ 失败: {title}")

            results.append(r)

        state.results = results
        progress(1.0, desc="全部完成！")
        return format_results_table(results), format_results_text(results), generate_export_json(results)

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
    return "<p style='color:#888; text-align:center; padding:20px;'>已清空，请上传视频文件</p>", "", None


# ========== Gradio 界面 ==========

# 界面文案常量
HEADER_MD = """
<div style="text-align:center; margin-bottom:8px;">
    <h1 style="margin:0; font-size:32px; font-weight:700;">🎬 短视频批量转文字</h1>
    <p style="margin:8px 0 0; color:#666; font-size:15px;">
        批量上传短视频文件，自动提取语音转写为文字。支持 MP4、MOV、AVI、WebM 等常见格式。
    </p>
    <p style="margin:4px 0 0; color:#999; font-size:13px;">
        视频时长建议 ≤ 90 秒，更长视频也能处理但耗时增加。
    </p>
</div>
"""

FAQ_MD = """
### 性能参考
| 模型 | 内存占用 | 90s视频转写耗时(CPU) | 中文准确率 |
|------|---------|---------------------|----------|
| tiny | ~1GB | ~15s | ~70% |
| base | ~1GB | ~30s | ~85% |
| small | ~2GB | ~60s | ~92% |
| large-v3 | ~3GB | ~180s | ~97% |

### 注意事项
- 首次运行会自动下载模型文件（base约74MB），后续不再重复下载
- CPU模式足够应对90s短视频，GPU可加速4-10倍
- 金融、证券专业术语建议用 base 或以上模型，tiny容易出错
- 如需更高准确率，可后续切换到腾讯云ASR引擎
"""

CUSTOM_CSS = """
/* 整体页面背景与内边距 */
.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    padding: 24px 16px 48px !important;
    display: flex !important;
    justify-content: center !important;
}

/* Gradio 6.x 主内容包裹层 */
.gradio-container > .main {
    width: 100% !important;
    max-width: 1000px !important;
    display: flex !important;
    justify-content: center !important;
}

/* 主内容区最大宽度并居中 */
.app-wrap {
    width: 100% !important;
    max-width: 1000px !important;
    margin: 0 auto !important;
}

/* 标题区卡片 */
.header-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    padding: 28px 24px;
    margin-bottom: 20px;
    text-align: center;
}

/* 上传区卡片 */
.main-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    padding: 24px;
    margin-bottom: 20px;
}

/* 结果区卡片 */
.result-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    padding: 24px;
    margin-bottom: 20px;
}

/* 按钮区域 */
.btn-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 8px;
}

/* 主按钮 */
.primary-btn {
    min-width: 180px;
}

/* 表格滚动 */
.result-table-wrap {
    overflow-x: auto;
}

/* 配置区紧凑 */
.config-row {
    margin-bottom: 8px !important;
}

/* 移动端适配 */
@media (max-width: 768px) {
    .gradio-container {
        padding: 12px 8px 32px !important;
    }
    .header-card, .main-card, .result-card {
        padding: 16px;
        border-radius: 12px;
    }
    .primary-btn {
        min-width: 100%;
    }
}
"""


def build_app():
    with gr.Blocks(
        title="短视频批量转文字",
        fill_width=False,
    ) as app:

        # 居中外层容器
        with gr.Column(elem_classes="app-wrap"):

            # 标题
            with gr.Column(elem_classes="header-card"):
                gr.Markdown(HEADER_MD)

            # 配置区
            with gr.Row(equal_height=True, elem_classes="config-row"):
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

            # 上传区
            with gr.Column(elem_classes="main-card"):
                file_input = gr.File(
                    label="上传短视频",
                    file_count="multiple",
                    file_types=[".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".3gp"],
                    type="filepath"
                )
                upload_btn = gr.Button(
                    "🚀 开始转写",
                    variant="primary",
                    size="lg",
                    elem_classes="primary-btn"
                )

            # 结果区
            with gr.Column(elem_classes="result-card"):
                gr.Markdown("## 📋 转写结果")

                result_table = gr.HTML(
                    value="<p style='color:#888; text-align:center; padding:20px;'>暂无结果，请上传视频文件</p>",
                    elem_classes="result-table-wrap"
                )

                result_text = gr.Textbox(
                    label="纯文本结果（可复制）",
                    value="",
                    lines=10,
                )

                # 导出区
                with gr.Row(elem_classes="btn-row"):
                    export_txt_btn = gr.Button("📄 导出 TXT")
                    export_json_btn = gr.Button("📦 导出 JSON")
                    export_excel_btn = gr.Button("📊 导出 Excel")
                    clear_btn = gr.Button("🗑️ 清空结果", variant="secondary")

                export_file = gr.File(label="导出文件")

            # 使用说明
            with gr.Column(elem_classes="main-card"):
                with gr.Accordion("📖 使用说明 & FAQ", open=False):
                    gr.Markdown(FAQ_MD)

        # 事件绑定
        upload_btn.click(
            fn=process_uploaded_files,
            inputs=[file_input, model_size, language],
            outputs=[result_table, result_text, export_file]
        )

        export_txt_btn.click(fn=export_txt_from_state, outputs=export_file)
        export_json_btn.click(fn=export_json_from_state, outputs=export_file)
        export_excel_btn.click(fn=export_excel_from_state, outputs=export_file)

        clear_btn.click(
            fn=clear_results,
            outputs=[result_table, result_text, export_file]
        )

    return app


# ========== 启动 ==========

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
        css=CUSTOM_CSS,
    )
