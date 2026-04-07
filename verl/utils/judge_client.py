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

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable


class JudgeClientError(RuntimeError):
    pass


@dataclass
class JudgeRequest:
    prompt: str
    response_text: str
    sentences: list[dict[str, Any]]
    overall_correct: bool
    buckets: list[float]
    extra: dict[str, Any] | None = None


class BaseJudgeClient:
    def __init__(self, rate_limit_qps: float | None = None) -> None:
        self._rate_limit_qps = float(rate_limit_qps or 0.0)
        self._last_call_s = 0.0

    def _sleep_for_rate_limit(self) -> None:
        if self._rate_limit_qps <= 0:
            return
        now = time.time()
        min_interval = 1.0 / self._rate_limit_qps
        elapsed = now - self._last_call_s
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_s = time.time()

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        raise NotImplementedError


class DummyJudgeClient(BaseJudgeClient):
    def __init__(self, buckets: list[float], rate_limit_qps: float | None = None) -> None:
        super().__init__(rate_limit_qps=rate_limit_qps)
        self._buckets = buckets

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        self._sleep_for_rate_limit()
        if not self._buckets:
            raise JudgeClientError("No buckets configured for dummy judge.")
        pos = self._buckets[1] if len(self._buckets) > 1 else self._buckets[0]
        neg = self._buckets[-2] if len(self._buckets) > 1 else -self._buckets[0]
        bucket = pos if request.overall_correct else neg
        sentences = [
            {"id": s["id"], "bucket": bucket, "confidence": 1.0, "reason": "dummy"}
            for s in request.sentences
        ]
        return {"sentences": sentences}


class CallableJudgeClient(BaseJudgeClient):
    def __init__(self, judge_fn: Callable[[dict[str, Any]], Any], rate_limit_qps: float | None = None) -> None:
        super().__init__(rate_limit_qps=rate_limit_qps)
        self._judge_fn = judge_fn

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        self._sleep_for_rate_limit()
        payload = {
            "prompt": request.prompt,
            "response_text": request.response_text,
            "sentences": request.sentences,
            "overall_correct": request.overall_correct,
            "buckets": request.buckets,
            "extra": request.extra or {},
        }
        raw = self._judge_fn(payload)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise JudgeClientError(f"Judge returned non-JSON: {raw[:200]}") from exc
        if isinstance(raw, dict):
            return raw
        raise JudgeClientError(f"Judge returned unsupported type: {type(raw)}")


def _import_from_path(path: str) -> Callable[[dict[str, Any]], Any]:
    if not path:
        raise JudgeClientError(f"Invalid judge_fn path: {path}")

    # Support both legacy "module.attr" and common "module:attr" syntaxes.
    if ":" in path:
        mod_name, attr = path.rsplit(":", 1)
    elif "." in path:
        mod_name, attr = path.rsplit(".", 1)
    else:
        raise JudgeClientError(f"Invalid judge_fn path: {path}")

    module = importlib.import_module(mod_name)
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise JudgeClientError(f"Judge function not found or not callable: {path}")
    return fn


def get_judge_client(cfg) -> BaseJudgeClient:
    backend = str(getattr(cfg, "judge_backend", "dummy") or "dummy").lower()
    rate_limit_qps = float(getattr(cfg, "rate_limit_qps", 0.0) or 0.0)
    buckets = list(getattr(cfg, "buckets", []))

    if backend == "dummy":
        return DummyJudgeClient(buckets=buckets, rate_limit_qps=rate_limit_qps)

    judge_fn_path = getattr(cfg, "judge_fn", None)
    if judge_fn_path:
        judge_fn = _import_from_path(str(judge_fn_path))
        return CallableJudgeClient(judge_fn=judge_fn, rate_limit_qps=rate_limit_qps)

    raise JudgeClientError(
        "Judge backend requires a callable judge_fn (e.g., 'my_pkg.my_module.my_judge')."
    )
