"""Custom reward wrappers for offline eval.

`main_eval` expects a callable with signature (data_source, response, ground_truth).
This module provides a simple default for math-style datasets (MATH/GSM8K/AIME).
"""
from __future__ import annotations

from typing import Any

from verl.utils.reward_score import gsm8k as gsm8k_score
from verl.utils.reward_score import math_reward
from verl.utils.reward_score import default_compute_score


def _unwrap_ground_truth(ground_truth: Any) -> str:
    """Accept dict or raw string and return the ground-truth answer string."""
    if isinstance(ground_truth, dict):
        if "ground_truth" in ground_truth:
            return ground_truth["ground_truth"]
        if "answer" in ground_truth:
            return ground_truth["answer"]
    return str(ground_truth)

def default_eval_fn(data_source: str, response: str, ground_truth: Any) -> float:
    """Use the same scoring function as training."""
    gt = _unwrap_ground_truth(ground_truth)
    
    # 直接调用训练时用的 default_compute_score
    score = default_compute_score(
        data_source=data_source,
        solution_str=response,
        ground_truth=gt,
        extra_info={},
    )
    
    if isinstance(score, dict):
        return float(score.get("score", 0.0))
    return float(score)
