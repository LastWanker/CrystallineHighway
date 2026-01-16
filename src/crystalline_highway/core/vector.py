"""向量工具函数。"""

from __future__ import annotations

import math
import random
from typing import List


def zero_vector(dim: int) -> List[float]:
    return [0.0 for _ in range(dim)]


def add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def scale(vec: List[float], s: float) -> List[float]:
    return [x * s for x in vec]


def mean(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0 for _ in range(dim)]
    for vec in vectors:
        for i, value in enumerate(vec):
            sums[i] += value
    return [value / len(vectors) for value in sums]


def distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def jitter(dim: int, scale_value: float) -> List[float]:
    """生成小扰动向量。"""

    return [random.uniform(-scale_value, scale_value) for _ in range(dim)]


def normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return list(vec)
    return [x / norm for x in vec]
