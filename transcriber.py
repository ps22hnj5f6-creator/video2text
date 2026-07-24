#!/usr/bin/env python3
"""视频转文字 - 核心转写引擎"""

import os

# HuggingFace 国内镜像（必须在 faster_whisper 导入之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptResult:
    """单个视频的转写结果"""
    source: str = ""          # 来源（文件路径或链接）
    title: str = ""           # 视频标题
    text: str = ""            # 完整转写文本
    segments: list = field(default_factory=list)  # 分段信息 [{start, end, text}]
    duration: float = 0.0     # 视频时长(秒)
    error: Optional[str] = None


def get_ffmpeg_path() -> str:
    """获取 ffmpeg 可执行文件路径"""
    # 优先用 imageio-ffmpeg 自带的
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    # 其次用系统 PATH 里的
    result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()

    raise RuntimeError(
        "ffmpeg 未找到！请安装：\n"
        "  macOS: brew install ffmpeg\n"
        "  Linux: apt install ffmpeg / yum install ffmpeg\n"
        "  或 pip install imageio-ffmpeg"
    )


def extract_audio(video_path: str, output_path: str = None, ffmpeg_path: str = None) -> str:
    """从视频中提取音频为 WAV 格式（16kHz mono，whisper最优输入）"""
    if ffmpeg_path is None:
        ffmpeg_path = get_ffmpeg_path()

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    cmd = [
        ffmpeg_path, "-i", video_path,
        "-vn",                # 不要视频
        "-acodec", "pcm_s16le",  # 16bit PCM
        "-ar", "16000",       # 16kHz采样率
        "-ac", "1",           # 单声道
        "-y",                 # 覆盖输出
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 音频提取失败: {result.stderr[:500]}")

    return output_path


def get_video_duration(video_path: str, ffmpeg_path: str = None) -> float:
    """获取视频时长"""
    if ffmpeg_path is None:
        ffmpeg_path = get_ffmpeg_path()

    cmd = [
        ffmpeg_path, "-i", video_path,
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    # 从 stderr 中提取时长
    for line in result.stderr.split("\n"):
        if "Duration:" in line:
            time_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = time_str.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    return 0.0


class WhisperTranscriber:
    """使用 faster-whisper 进行语音转写"""

    # 中文金融场景的提示词 —— 让 Whisper 输出标点 + 减少金融术语错别字
    ZH_FINANCE_PROMPT = (
        "以下是普通话的句子，带有标点符号。"
        "对于股票、基金、期货、债券、证券、A股、港股、美股、"
        "涨停、跌停、大盘、创业板、科创板、北交所、"
        "成交量、换手率、均线、K线、MACD、市盈率、市净率、"
        "打板、龙头、连板、板块、轮动、题材、热点、"
        "仓位、止损、止盈、融资、融券、量化、打新、"
        "北向资金、南向资金、外资、内资、机构、游资、"
        "分红、除权、除息、复权、龙虎榜、大宗交易、"
        "沪深300、中证500、上证50、上证指数、深证成指、"
        "注册制、退市、IPO、定增、配股、转债、"
        "基金经理、基金经理、投顾、券商、银行、保险、"
        "宏观数据、GDP、CPI、PMI、M2、降息、加息、"
        "请准确识别以上金融术语，不要写错。"
    )

    # 常见 Whisper 中文错别字纠正映射
    ZH_CORRECTION_MAP = {
        "脱骨": "脱手", "脱姑": "脱手",
        "涨停版": "涨停板", "跌停版": "跌停板",
        "大版": "大盘", "小版": "小盘",
        "创越版": "创业板", "科创版": "科创板",
        "上正": "上证", "深正": "深证",
        "市营率": "市盈率", "市静率": "市净率",
        "换手绿": "换手率", "成交绿": "成交量",
        "均限": "均线", "K限": "K线",
        "北交锁": "北交所", "北交所": "北交所",
        "融止": "融资", "容资": "融资",
        "融卷": "融券", "容券": "融券",
        "量话": "量化", "量划": "量化",
        "打新版": "打新股", "打板版": "打板",
        "龙虎榜版": "龙虎榜",
        "止营": "止盈", "只赢": "止盈",
        "只损": "止损", "止笋": "止损",
        "除息权": "除权除息",
        "复权权": "复权",
        "配骨": "配股", "定增增": "定增",
        "转卷": "转债", "转债债": "转债",
        "基金经里": "基金经理",
        "投故": "投顾", "投顾顾": "投顾",
        "券梢": "券商", "券商商": "券商",
        "降息息": "降息", "加息息": "加息",
        "大宗交宜": "大宗交易",
        "注册制制": "注册制",
        "退市市": "退市",
        "IPOO": "IPO",
        "GDPD": "GDP", "CPIC": "CPI", "PMIP": "PMI",
    }

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8", language: str = "zh"):
        """
        Args:
            model_size: 模型大小 tiny/base/small/medium/large-v3
                       tiny=39M, base=74M, small=244M, medium=769M, large-v3=1550M
                       base 是性价比最佳选择（中文金融场景足够）
            device: cpu 或 cuda
            compute_type: int8(float16在CPU上不支持)
            language: zh=中文, en=英文, None=自动检测
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _load_model(self):
        """延迟加载模型（首次调用时才下载+加载）"""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
            print(f"正在加载 faster-whisper 模型 ({self.model_size})...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            print("模型加载完成")
        except ImportError:
            raise ImportError(
                "faster-whisper 未安装！请运行：pip install faster-whisper\n"
                "如需 GPU 加速，还需：pip install ctranslate2[cuda]"
            )

    def _correct_typos(self, text: str) -> str:
        """纠正 Whisper 中文常见错别字"""
        for wrong, correct in self.ZH_CORRECTION_MAP.items():
            text = text.replace(wrong, correct)
        # 清理多余重复标点
        import re
        text = re.sub(r'([。！？])\1+', r'\1', text)  # 句号/叹号/问号不重复
        text = re.sub(r'([，])\1+', r'\1', text)       # 逗号不重复
        return text

    def transcribe_file(self, video_path: str, title: str = "") -> TranscriptResult:
        """转写单个视频文件"""
        result = TranscriptResult(source=video_path, title=title)

        try:
            # 1. 获取时长
            result.duration = get_video_duration(video_path)

            # 2. 提取音频
            audio_path = extract_audio(video_path)

            # 3. 加载模型
            self._load_model()

            # 4. 转写（使用金融提示词 + 更严格解码参数）
            print(f"正在转写: {title or video_path}")

            # 根据语言选择提示词
            initial_prompt = None
            if self.language == "zh" or self.language is None:
                initial_prompt = self.ZH_FINANCE_PROMPT

            segments_iter, info = self._model.transcribe(
                audio_path,
                language=self.language,
                initial_prompt=initial_prompt,  # 提示词：输出标点 + 金融术语
                beam_size=5,                     # 更准确的解码
                best_of=5,                       # 保留最优候选
                temperature=0.0,                 # 低温度=更准确、少幻觉
                condition_on_previous_text=True,  # 利用上文减少重复
                no_speech_threshold=0.6,         # 更严格的静音检测
                vad_filter=True,                 # VAD过滤静音段
                vad_parameters=dict(
                    min_silence_duration_ms=300,  # 300ms静音即分段（标点依据）
                    threshold=0.5
                ),
                word_timestamps=True,            # 词级时间戳（更精确对齐）
            )

            # 5. 收集结果 + 加标点
            full_text = ""
            segment_list = []
            prev_end = 0.0

            for seg in segments_iter:
                text = seg.text.strip()

                # 根据段间停顿自动加标点
                gap = seg.start - prev_end
                if prev_end > 0 and gap > 1.5 and text:
                    # 停顿>1.5秒 → 加句号
                    if full_text and not full_text.endswith(("。", "！", "？", ".", "!", "?", "，", ",")):
                        full_text += "。"
                elif prev_end > 0 and gap > 0.6 and text:
                    # 停顿0.6-1.5秒 → 加逗号
                    if full_text and not full_text.endswith(("。", "！", "？", ".", "!", "?", "，", ",")):
                        full_text += "，"

                full_text += text
                prev_end = seg.end

                segment_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text
                })

            # 句末补句号
            if full_text and not full_text.endswith(("。", "！", "？", ".", "!", "?")):
                full_text += "。"

            # 6. 错别字纠正
            full_text = self._correct_typos(full_text)
            for seg in segment_list:
                seg["text"] = self._correct_typos(seg["text"])

            result.text = full_text
            result.segments = segment_list

            # 6. 清理临时音频文件
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        except Exception as e:
            result.error = str(e)

        return result

    def transcribe_batch(self, video_paths: list[str], titles: list[str] = None) -> list[TranscriptResult]:
        """批量转写多个视频"""
        if titles is None:
            titles = [Path(p).stem for p in video_paths]

        results = []
        for i, (path, title) in enumerate(zip(video_paths, titles)):
            print(f"\n[{i+1}/{len(video_paths)}] 处理: {title}")
            r = self.transcribe_file(path, title)
            results.append(r)
            if r.error:
                print(f"  ❌ 失败: {r.error}")
            else:
                print(f"  ✅ 完成, 时长={r.duration:.1f}s, 字数={len(r.text)}")

        return results


# ========== 备用方案：腾讯云 ASR ==========

class TencentASRTranscriber:
    """使用腾讯云一句话识别 API（适合 60s 以内音频）
    注意：一句话识别上限60s，如果视频>60s需要先用长语音识别
    """

    def __init__(self, secret_id: str = "", secret_key: str = "",
                 engine_model_type: str = "16k_zh"):
        self.secret_id = secret_id or os.getenv("TENCENT_SECRET_ID", "")
        self.secret_key = secret_key or os.getenv("TENCENT_SECRET_KEY", "")
        self.engine_model_type = engine_model_type
        self._ffmpeg_path = None

    def transcribe_file(self, video_path: str, title: str = "") -> TranscriptResult:
        """使用腾讯云 ASR 转写"""
        result = TranscriptResult(source=video_path, title=title)

        if not self.secret_id or not self.secret_key:
            result.error = "腾讯云密钥未设置，请设置 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY"
            return result

        try:
            import base64
            import json
            import hashlib
            import hmac
            import time
            from datetime import datetime

            # 提取音频
            audio_path = extract_audio(video_path)
            result.duration = get_video_duration(video_path)

            # 读取音频数据
            with open(audio_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")

            # 腾讯云一句话识别 API
            # 这里仅作示例，实际需要完整的签名逻辑
            # 推荐用 tencentcloud-sdk-python 包
            try:
                from tencentcloud.common import credential
                from tencentcloud.asr.v20190614 import models, client

                cred = credential.Credential(self.secret_id, self.secret_key)
                cli = client.AsrClient(cred, "ap-beijing")

                req = models.CreateRecTaskRequest()
                req.EngineModelType = self.engine_model_type
                req.ChannelNum = 1
                req.SourceType = 1  # 音频URL方式
                req.Data = audio_data
                req.DataLen = len(audio_data)
                req.ResTextFormat = 3  # 含时间戳

                resp = cli.CreateRecTask(req)
                task_id = resp.Data.TaskId

                # 證查询结果
                for _ in range(30):
                    desc_req = models.DescribeTaskStatusRequest()
                    desc_req.TaskId = task_id
                    desc_resp = cli.DescribeTaskStatus(desc_req)
                    if desc_resp.Data.StatusStr == "success":
                        result.text = desc_resp.Data.Result
                        break
                    elif desc_resp.Data.StatusStr == "failed":
                        result.error = desc_resp.Data.ErrorMsg
                        break
                    time.sleep(2)

            except ImportError:
                result.error = "请安装腾讯云SDK: pip install tencentcloud-sdk-python"

            # 清理
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        except Exception as e:
            result.error = str(e)

        return result


# ========== 工厂函数 ==========

def create_transcriber(engine: str = "whisper", **kwargs):
    """创建转写引擎实例

    Args:
        engine: "whisper" 或 "tencent_asr"
        kwargs: 传递给对应引擎的参数
    """
    if engine == "whisper":
        return WhisperTranscriber(**kwargs)
    elif engine == "tencent_asr":
        return TencentASRTranscriber(**kwargs)
    else:
        raise ValueError(f"未知引擎: {engine}, 支持: whisper, tencent_asr")
