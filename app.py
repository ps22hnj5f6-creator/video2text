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
from deepseek_cleaner import clean_results_with_deepseek


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

def _progress_html(step: int, total: int, desc: str, done: bool = False) -> str:
    """渲染单条自定义进度条 HTML"""
    pct = 100 if done else (min(step / total, 1.0) * 100 if total else 0)
    color = "#22c55e" if done else "#4f46e5"
    desc_safe = desc.replace("\"", "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <div class="v2t-progress">
        <div class="v2t-progress-header">
            <span class="v2t-progress-desc">{desc_safe}</span>
            <span class="v2t-progress-pct">{step}/{total}</span>
        </div>
        <div class="v2t-progress-track">
            <div class="v2t-progress-bar" style="width:{pct:.1f}%; background:{color};"></div>
        </div>
    </div>
    """


def _results_to_state(results: list) -> list[dict]:
    """把 TranscriptResult 列表序列化为可在 gr.State 中传递的纯字典列表"""
    return [
        {
            "source": r.source,
            "title": r.title,
            "duration": r.duration,
            "text": r.text,
            "segments": r.segments,
            "error": r.error,
        }
        for r in results
    ]


def process_uploaded_files_streaming(files, model_size, language, use_deepseek, deepseek_api_key):
    """处理本地上传的视频文件（流式返回状态，只渲染一条进度条）"""
    results = []
    try:
        if not files:
            yield _progress_html(0, 1, "请上传至少一个视频文件", done=True), []
            return

        need_reload = (state.model_size != model_size or state.language != language)
        state.model_size = model_size
        state.language = language
        if need_reload:
            state.transcriber = None

        if isinstance(files, list):
            video_paths = [f if isinstance(f, str) else getattr(f, 'name', str(f)) for f in files]
        elif isinstance(files, str):
            video_paths = [files]
        else:
            video_paths = [getattr(files, 'name', str(files))]

        titles = [Path(p).stem for p in video_paths]
        total_files = len(video_paths)
        model_load_step = 1 if state.transcriber is None else 0
        deepseek_step = 1 if use_deepseek else 0
        total_steps = total_files * 3 + model_load_step + deepseek_step

        # 准备阶段
        yield _progress_html(0, total_steps, "正在准备..."), _results_to_state(results)

        # 首次加载模型
        if model_load_step:
            yield _progress_html(1, total_steps, "正在加载 Whisper 模型（首次需下载）..."), _results_to_state(results)

        transcriber = state.get_transcriber()

        current = 1 + model_load_step
        for i, (path, title) in enumerate(zip(video_paths, titles)):
            r = TranscriptResult(source=path, title=title)

            yield _progress_html(current, total_steps, f"[{i+1}/{total_files}] 提取音频: {title}"), _results_to_state(results)
            current += 1

            try:
                audio_path, duration = transcriber.prepare_audio(path)
                r.duration = duration

                yield _progress_html(current, total_steps, f"[{i+1}/{total_files}] 正在转写: {title}"), _results_to_state(results)
                current += 1

                full_text, segment_list = transcriber.transcribe_audio(audio_path)
                r.text = full_text
                r.segments = segment_list

                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

                yield _progress_html(current, total_steps, f"[{i+1}/{total_files}] ✅ 完成: {title}"), _results_to_state(results)
                current += 1

            except Exception as e:
                r.error = str(e)
                current += 2
                yield _progress_html(
                    min(current, total_steps), total_steps,
                    f"[{i+1}/{total_files}] ❌ 失败: {title} — {e}"
                ), _results_to_state(results)

            results.append(r)
            state.results = results

        if use_deepseek:
            yield _progress_html(current, total_steps, "🧹 DeepSeek 正在清理重复与错词..."), _results_to_state(results)
            current += 1

            api_key = (deepseek_api_key or "").strip()
            if not api_key or api_key.startswith("$"):
                api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if api_key:
                try:
                    results = clean_results_with_deepseek(results, api_key)
                    for r in results:
                        if r.segments and r.text:
                            for seg in r.segments:
                                seg["text"] = r.text
                    state.results = results
                except Exception as e:
                    err_msg = f"DeepSeek 后处理失败: {e}"
                    print(f"[WARN] {err_msg}")
                    traceback.print_exc()
                    yield _progress_html(total_steps, total_steps, f"⚠️ {err_msg}", done=True), _results_to_state(results)
                    return
            else:
                yield _progress_html(
                    total_steps, total_steps,
                    "⚠️ 已启用 DeepSeek 但 API key 为空（未设置 DEEPSEEK_API_KEY）",
                    done=True
                ), _results_to_state(results)
                return

        yield _progress_html(total_steps, total_steps, "✅ 全部完成！", done=True), _results_to_state(results)

    except Exception as e:
        traceback.print_exc()
        yield _progress_html(0, 1, f"❌ 处理出错: {e}", done=True), _results_to_state(results)


def render_results(_state_data):
    """根据 state.results 渲染最终结果区（由 .then() 调用）"""
    results = state.results
    if not results:
        return "<p style='color:#888; text-align:center; padding:20px;'>暂无结果</p>", "", None
    return format_results_table(results), format_results_text(results), generate_export_json(results)


# ========== 结果格式化 ==========

def format_results_table(results: list) -> str:
    if not results:
        return "<p style='color:#888; text-align:center; padding:20px;'>暂无结果</p>"

    total = len(results)
    success = sum(1 for r in results if not r.error)
    failed = total - success

    # 紧凑表头统计
    html = f"""
    <div class="result-summary">
        共 {total} 条 | ✅ 成功 {success} 条 | ❌ 失败 {failed} 条
    </div>
    <div class="result-table-inner">
    <table class="compact-result-table">
    <thead>
    <tr>
        <th style="width:28px;">#</th>
        <th style="width:25%;">标题</th>
        <th style="width:45px;">时长</th>
        <th style="width:45px;">字数</th>
        <th>内容预览</th>
        <th style="width:36px;">状态</th>
    </tr>
    </thead>
    <tbody>
    """

    for i, r in enumerate(results):
        status = "✅" if not r.error else "❌"
        if r.error:
            content = f"<span style='color:#dc2626; font-size:12px;'>{r.error}</span>"
        else:
            content = r.text[:80] + "..." if len(r.text) > 80 else (r.text or "-")
            # 高亮常见金融关键词
            content = _highlight_finance_terms(content)

        duration_str = f"{r.duration:.1f}s" if r.duration > 0 else "-"
        word_count = len(r.text) if r.text else 0
        source = r.title or r.source[:50]

        row_class = "error" if r.error else ""
        html += f"""
    <tr class="{row_class}">
        <td class="idx">{i+1}</td>
        <td class="title" title="{source}">{source}</td>
        <td class="num">{duration_str}</td>
        <td class="num">{word_count}</td>
        <td class="preview">{content}</td>
        <td class="status">{status}</td>
    </tr>"""

    html += """
    </tbody>
    </table>
    </div>
    """
    return html


def _highlight_finance_terms(text: str) -> str:
    """在预览中高亮常见金融术语，便于快速扫读"""
    import re
    terms = [
        "主线", "资金", "扎堆", "板块", "版块", "科技", "医药", "券商", "消费",
        "满仓", "空仓", "半仓", "止损", "止盈", "融资", "融券", "量化",
        "大盘", "创业板", "科创板", "北交所", "A股", "港股", "美股",
        "涨停", "跌停", "拉升", "回调", "调整", "震荡", "放量", "缩量",
        "尾盘", "开盘", "收盘", "成交量", "换手率", "K线", "均线",
        "北向资金", "南向资金", "机构", "游资", "主力", "外资", "内资",
        "龙虎榜", "大宗交易", "IPO", "定增", "配股", "转债",
    ]
    pattern = "|".join(re.escape(t) for t in terms)
    if not pattern:
        return text
    return re.sub(
        f"({pattern})",
        r'<span class="fin-term">\1</span>',
        text
    )


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
    return (
        "<p style='color:#888; text-align:center; padding:20px;'>已清空，请上传视频文件</p>",
        "",
        None,
        "",
    )


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

/* 紧凑结果表 */
.result-summary {
    text-align: center;
    color: #666;
    font-size: 13px;
    margin-bottom: 10px;
}

.result-table-inner {
    max-height: 360px;
    overflow-y: auto;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    background: #fff;
}

.compact-result-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.compact-result-table thead {
    position: sticky;
    top: 0;
    background: #f8fafc;
    z-index: 1;
}

.compact-result-table th {
    padding: 8px 6px;
    font-weight: 600;
    color: #475569;
    border-bottom: 1px solid #e2e8f0;
    text-align: left;
    white-space: nowrap;
}

.compact-result-table td {
    padding: 6px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: top;
}

.compact-result-table tbody tr:last-child td {
    border-bottom: none;
}

.compact-result-table tbody tr:nth-child(even) {
    background: #fafafa;
}

.compact-result-table tbody tr.error {
    background: #fef2f2;
}

.compact-result-table td.idx,
.compact-result-table td.num,
.compact-result-table td.status {
    text-align: center;
    color: #64748b;
}

.compact-result-table td.title {
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.compact-result-table td.preview {
    color: #334155;
    line-height: 1.5;
}

.compact-result-table .fin-term {
    color: #0369a1;
    font-weight: 600;
    background: #f0f9ff;
    padding: 1px 3px;
    border-radius: 3px;
}

/* 导出文件组件 */
.export-file .file-preview {
    margin-top: 8px;
}

/* 配置区紧凑 */
.config-row {
    margin-bottom: 8px !important;
}

/* 单一状态/进度条区域 */
.status-box {
    margin-bottom: 16px;
}

.v2t-progress {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

.v2t-progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    font-size: 14px;
    color: #334155;
}

.v2t-progress-desc {
    font-weight: 500;
}

.v2t-progress-pct {
    color: #64748b;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
}

.v2t-progress-track {
    width: 100%;
    height: 10px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
}

.v2t-progress-bar {
    height: 100%;
    border-radius: 999px;
    transition: width 0.25s ease;
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

            # DeepSeek 后处理配置（默认开启，key 由部署环境变量提供）
            with gr.Column(elem_classes="main-card"):
                with gr.Row(equal_height=True):
                    use_deepseek = gr.Checkbox(
                        label="✨ 使用 DeepSeek 后处理",
                        value=True,
                        info="清理末尾重复、纠正同音错别字和金融术语（默认开启）",
                    )
                    deepseek_api_key = gr.Textbox(
                        label="DeepSeek API Key",
                        placeholder="sk-...  或留空，自动使用服务器环境变量 DEEPSEEK_API_KEY",
                        value=os.getenv("DEEPSEEK_API_KEY", ""),
                        type="password",
                        max_lines=1,
                        show_label=True,
                        info="部署端已内嵌默认 Key，本地运行可在此覆盖或留空用环境变量",
                    )
                gr.Markdown(
                    "<p style='margin:0; color:#888; font-size:12px;'>"
                    "没有 Key？前往 <a href='https://platform.deepseek.com' target='_blank'>DeepSeek 开放平台</a> 申请；"
                    "线上部署已通过环境变量 <code>DEEPSEEK_API_KEY</code> 内嵌默认 Key，开箱即用。</p>"
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

                # 单一状态/进度条（替代 Gradio 多输出进度条）
                status_box = gr.HTML(
                    value="<p style='color:#888; text-align:center; padding:12px;'>点击「开始转写」后在此显示进度</p>",
                    elem_classes="status-box"
                )
                # 隐藏状态，用于触发最终渲染
                results_state = gr.State(value=[])

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
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")

                export_file = gr.File(label="导出文件", elem_classes="export-file")

            # 使用说明
            with gr.Column(elem_classes="main-card"):
                with gr.Accordion("📖 使用说明 & FAQ", open=False):
                    gr.Markdown(FAQ_MD)

        # 事件绑定
        # 流式处理：只更新单一状态区，隐藏 Gradio 默认的多输出进度条
        upload_btn.click(
            fn=process_uploaded_files_streaming,
            inputs=[file_input, model_size, language, use_deepseek, deepseek_api_key],
            outputs=[status_box, results_state],
            show_progress="hidden",
        ).then(
            fn=render_results,
            inputs=[results_state],
            outputs=[result_table, result_text, export_file]
        )

        export_txt_btn.click(fn=export_txt_from_state, outputs=export_file)
        export_json_btn.click(fn=export_json_from_state, outputs=export_file)
        export_excel_btn.click(fn=export_excel_from_state, outputs=export_file)

        clear_btn.click(
            fn=clear_results,
            outputs=[result_table, result_text, export_file, status_box]
        )

    return app


# ========== 启动 ==========

if __name__ == "__main__":
    app = build_app()
    # 启用队列：长转写任务用持久连接，避免代理/负载均衡超时断开
    app.queue(
        default_concurrency_limit=1,
        max_size=20,
        status_update_rate="auto",
    )
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
