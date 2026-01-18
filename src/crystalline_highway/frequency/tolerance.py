"""频率与容忍度映射规则。"""

from __future__ import annotations

import statistics
from typing import Iterable

from .calibration import FrequencyCalibration


def private_typical(private_counts: Iterable[float], default: float) -> float:
    values = [count for count in private_counts if count > 0]
    if not values:
        return default
    return float(statistics.median(values))


def private_boost(private_count: float, typical: float, beta: float, cap: float) -> float:
    ratio = private_count / max(typical, 1.0)
    boost = (1.0 + ratio) ** beta
    return min(boost, cap)


def effective_frequency(
    global_freq: float,
    private_count: float,
    typical: float,
    beta: float,
    cap: float,
    eps: float,
) -> float:
    boost = private_boost(private_count, typical, beta, cap)
    return max(global_freq, eps) * boost


def radius_from_frequency(
    calibration: FrequencyCalibration,
    effective_freq: float,
    hit_probability: float,
    avg_freq: float,
    radius_min: float,
    radius_max: float,
    eps: float,
) -> float:
    distance_scale = calibration.distance_scale(hit_probability)
    radius = distance_scale * (avg_freq / max(effective_freq, eps))
    if radius < radius_min:
        return radius_min
    if radius > radius_max:
        return radius_max
    return radius
