#!/usr/bin/env python3
"""DeepSeek API 后处理清理模块

Whisper 初稿常见问题：
- 末尾静音导致循环重复（如"体体体体体..."）
- 同音错别字（围盘→尾盘、版块→板块、卖人一步→慢人一步）
- 财经术语识别偏差

本模块把多条转写文本批量送给 DeepSeek 做清理和纠错，返回结构化结果。
API key 由调用方传入，本模块不读取任何环境变量/配置文件，避免泄露。
"""

import os
import json
import re
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class CleanResult:
    index: int
    original_text: str
    cleaned_text: str
    error: Optional[str] = None


class DeepSeekCleaner:
    """DeepSeek 文本后处理清理器"""

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"

    SYSTEM_PROMPT = """你是一名财经短视频字幕校对专家。我会给你若干条语音识别初稿，它们来自普通话财经直播短视频，可能存在以下问题：
1. 末尾因静音导致的循环重复（如"体体体体体..."或"我们7月7日"反复出现多次）
2. 同音错别字，常见错误对照：
   - 围盘/委盘/维盘 → 尾盘
   - 版块/板快 → 板块
   - 卖人一步 → 慢人一步
   - 紫丹 → 子弹
   - 大低 → 大跌
   - 部长（在"把...做好"语境中） → 部署/底部
   - 吃向 → 吃香/吃相
   - 主现 → 主线
   - 冲近来/冲进莱 → 冲进来
   - 均限 → 均线，K限 → K线
   - 市营率 → 市盈率，市静率 → 市净率
   - 换手绿 → 换手率，成交绿 → 成交量
3. 金融证券专业术语识别错误

请对每条进行清理和纠错：
- 删除无意义的循环重复字符或句子
- 把同音错别字、口语错词改为正确表达
- 保留原意和口语风格，不要过度润色
- 不要添加原文中没有的信息
- 不要改动没有错的内容

请严格按以下 JSON 格式返回，不要返回 markdown 代码块，不要返回任何解释：
{
  "results": [
    {"index": 0, "cleaned_text": "第一条清理后的文本"},
    {"index": 1, "cleaned_text": "第二条清理后的文本"}
  ]
}"""

    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 120.0):
        if not api_key or not api_key.strip():
            raise ValueError("DeepSeek API key 不能为空")
        self.api_key = api_key.strip()
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """延迟初始化 OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "使用 DeepSeek 后处理需要安装 openai 包：pip install openai"
                ) from e
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    @staticmethod
    def _build_user_message(texts: list[str]) -> str:
        lines = []
        for i, text in enumerate(texts):
            # 先做一层简单转义，防止破坏 prompt 结构
            safe_text = text.replace("\"", "'")
            lines.append(f"[{i}] {safe_text}")
        return "\n\n".join(lines)

    def clean_batch(self, texts: list[str]) -> list[CleanResult]:
        """批量清理文本

        Args:
            texts: 原始转写文本列表

        Returns:
            CleanResult 列表。如果 API 调用失败，每条都返回原文字并附带错误信息。
        """
        if not texts:
            return []

        # 记录原始非空索引，空文本直接跳过 API 调用
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty_indices:
            return [CleanResult(index=i, original_text=t, cleaned_text=t)
                    for i, t in enumerate(texts)]

        non_empty_texts = [texts[i] for i in non_empty_indices]

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_user_message(non_empty_texts)},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)

            # 建立 index -> cleaned_text 映射
            cleaned_map = {}
            if isinstance(parsed, dict) and "results" in parsed:
                for item in parsed["results"]:
                    idx = item.get("index")
                    cleaned = item.get("cleaned_text", "")
                    if idx is not None and 0 <= idx < len(non_empty_texts):
                        cleaned_map[idx] = cleaned

            # 组装结果
            results = []
            for i, original in enumerate(texts):
                if i not in non_empty_indices:
                    results.append(CleanResult(
                        index=i, original_text=original, cleaned_text=original
                    ))
                    continue

                local_idx = non_empty_indices.index(i)
                cleaned = cleaned_map.get(local_idx, original)
                results.append(CleanResult(
                    index=i, original_text=original, cleaned_text=cleaned
                ))
            return results

        except Exception as e:
            # 兜底：API 失败时原样返回
            err_msg = f"DeepSeek 清理失败: {str(e)[:200]}"
            traceback.print_exc()
            results = []
            for i, original in enumerate(texts):
                if i not in non_empty_indices:
                    results.append(CleanResult(
                        index=i, original_text=original, cleaned_text=original
                    ))
                else:
                    results.append(CleanResult(
                        index=i, original_text=original,
                        cleaned_text=original, error=err_msg
                    ))
            return results

    def clean_one(self, text: str) -> CleanResult:
        """单条文本清理"""
        results = self.clean_batch([text])
        return results[0] if results else CleanResult(
            index=0, original_text=text, cleaned_text=text,
            error="DeepSeek 返回为空"
        )


def clean_results_with_deepseek(
    results: list,
    api_key: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> list:
    """ Convenience 函数：直接清理 TranscriptResult 列表并原地更新 text 字段

    Args:
        results: TranscriptResult 列表
        api_key: DeepSeek API key
        base_url: 可选，自定义 API 地址
        model: 可选，自定义模型

    Returns:
        更新后的 results 列表（同一个对象）
    """
    if not api_key:
        return results

    cleaner = DeepSeekCleaner(api_key=api_key, base_url=base_url, model=model)
    texts = [getattr(r, "text", "") for r in results]
    cleaned = cleaner.clean_batch(texts)

    for r, c in zip(results, cleaned):
        if c.cleaned_text and c.cleaned_text.strip():
            r.text = c.cleaned_text
        if c.error:
            # 不影响主流程，只在原文末尾追加提示
            r.error = (r.error or "") + f" [{c.error}]"

    return results
