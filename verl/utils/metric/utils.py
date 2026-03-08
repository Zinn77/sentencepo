# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics utils.
"""

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some envs
    torch = None


def _to_scalar_list(val: list[Any]) -> list[float]:
    scalars: list[float] = []
    for v in val:
        if v is None:
            continue
        if torch is not None and isinstance(v, torch.Tensor):
            if v.numel() == 0:
                continue
            if hasattr(torch, "nanmean"):
                scalars.append(float(torch.nanmean(v.detach().float()).item()))
            else:
                scalars.append(float(v.detach().float().mean().item()))
        elif isinstance(v, np.ndarray):
            if v.size == 0:
                continue
            scalars.append(float(np.nanmean(v)))
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                continue
            scalars.append(float(np.nanmean(v)))
        else:
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                continue
            scalars.append(float(v))
    return scalars


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        val_list = _to_scalar_list(val)
        if not val_list:
            metrics[key] = np.nan
            continue
        if "max" in key:
            metrics[key] = np.max(val_list)
        elif "min" in key:
            metrics[key] = np.min(val_list)
        else:
            metrics[key] = np.mean(val_list)
    return metrics
