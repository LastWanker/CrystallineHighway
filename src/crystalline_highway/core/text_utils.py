"""文本规范化与标点处理工具。"""

from __future__ import annotations

import re

# 常见中英文标点符号集合，用于“寻找时忽略标点”的规则。
PUNCTUATION_PATTERN = re.compile(
    r"[\s\u3000"
    r"。！？!?；;，,、：:「」『』“”\"'（）\(\)【】\[\]《》<>"
    r"…—\-·]"
)


def normalize_text(text: str) -> str:
    """规范化文本，用于字典键与搜索输入。

    规则：删除空白与标点，保留核心字序。
    这是为了满足“寻找时忽略标点”的要求，避免标点造成重复词条。
    """

    if not text:
        return ""
    return PUNCTUATION_PATTERN.sub("", text)


def strip_punctuation(text: str) -> str:
    """移除文本中的标点与空白，适合用于分词前的净化。"""

    return normalize_text(text)


def is_punctuation(text: str) -> bool:
    """判断一段文本是否只包含标点/空白。"""

    return not normalize_text(text)
