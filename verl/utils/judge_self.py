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

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in judge response.")


def judge_fn(payload: dict[str, Any]) -> dict[str, Any]:
    """Self-judge callable for Sentence Judge Advantage.

    Expects payload from CallableJudgeClient. Sends an OpenAI-compatible
    chat completion request to a local vLLM server and returns parsed JSON.
    """

    base_url = os.environ.get("SJA_JUDGE_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.environ.get("SJA_JUDGE_MODEL", "")
    if not model:
        raise RuntimeError("SJA_JUDGE_MODEL is required for judge_fn.")

    prompt = payload.get("extra", {}).get("rendered_prompt") or payload.get("prompt")
    if not prompt:
        raise RuntimeError("judge_fn requires rendered_prompt or prompt in payload.")

    temperature = float(os.environ.get("SJA_JUDGE_TEMPERATURE", "0.0"))
    max_tokens = int(os.environ.get("SJA_JUDGE_MAX_TOKENS", "512"))
    timeout_s = int(os.environ.get("SJA_JUDGE_TIMEOUT_S", "60"))
    retries = int(os.environ.get("SJA_JUDGE_RETRIES", "2"))

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = _post_json(f"{base_url}/chat/completions", body, timeout_s=timeout_s)
            content = resp["choices"][0]["message"]["content"]
            return _extract_json(content)
        except (KeyError, IndexError, ValueError, urllib.error.URLError) as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"judge_fn failed after retries: {last_err}")
