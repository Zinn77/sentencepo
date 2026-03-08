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
from typing import Any

import torch

from verl.utils.judge_client import BaseJudgeClient, JudgeRequest, get_judge_client


class SentenceJudgeAdvantage:
    def __init__(self, cfg, tokenizer, device, judge_client: BaseJudgeClient | None = None) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device
        backend = str(getattr(cfg, "judge_backend", "dummy") or "dummy").lower()
        if judge_client is not None:
            self.judge_client = judge_client
        elif backend in {"self", "actor"}:
            self.judge_client = None
        else:
            self.judge_client = get_judge_client(cfg)
        self.buckets = [float(b) for b in getattr(cfg, "buckets", [])]

    def _build_prompt(self, prompt: str, sentence_texts: list[str], overall_correct: bool) -> str:
        buckets = ", ".join([str(b) for b in self.buckets])
        numbered = "\n".join([f"S{i + 1}: {s}" for i, s in enumerate(sentence_texts)])
        return (
            "You are a sentence-level judge. "
            "Score each sentence with a discrete advantage bucket and a confidence in [0,1].\n"
            f"Allowed buckets: [{buckets}].\n"
            f"overall_correct = {str(bool(overall_correct)).lower()}\n\n"
            "PROMPT:\n"
            f"{prompt}\n\n"
            "RESPONSE (sentence numbered):\n"
            f"{numbered}\n\n"
            "Output strict JSON only:\n"
            "{\"sentences\":[{\"id\":1,\"bucket\":0.25,\"confidence\":0.8,\"reason\":\"...\"}]}"
        )

    def _normalize_sentence_adv(self, adv: list[float]) -> list[float]:
        if not adv:
            return adv
        if str(getattr(self.cfg, "normalize", "none")).lower() != "zscore":
            return adv
        mean = sum(adv) / len(adv)
        var = sum((x - mean) ** 2 for x in adv) / max(len(adv), 1)
        std = (var ** 0.5) if var > 0 else 1.0
        return [(x - mean) / std for x in adv]

    def _apply_confidence(self, adv: list[float], conf: list[float]) -> list[float]:
        use_weight = bool(getattr(self.cfg, "use_confidence_weight", True))
        floor = float(getattr(self.cfg, "confidence_floor", 0.2))
        out: list[float] = []
        for a, c in zip(adv, conf, strict=False):
            c = float(min(max(c, 0.0), 1.0))
            if c < floor:
                out.append(0.0)
                continue
            out.append(a * c if use_weight else a)
        return out

    def _enforce_consistency(self, adv: list[float], overall_correct: bool) -> tuple[list[float], bool]:
        if not adv:
            return adv, False
        mean = sum(adv) / len(adv)
        sign = 1.0 if overall_correct else -1.0
        fixed = False
        if sign * mean < 0:
            adv = [a - mean for a in adv]
            fixed = True
        if sign * sum(adv) < 0:
            adv = [-a for a in adv]
            fixed = True
        return adv, fixed

    def _parse_judge_output(self, raw: dict[str, Any], num_sentences: int) -> tuple[list[float], list[float]]:
        sentences = raw.get("sentences", [])
        if len(sentences) != num_sentences:
            raise ValueError("Judge output missing sentences or count mismatch.")
        bucket_set = set(self.buckets)
        adv = [0.0] * num_sentences
        conf = [0.0] * num_sentences
        for item in sentences:
            idx = int(item.get("id", 0)) - 1
            if idx < 0 or idx >= num_sentences:
                continue
            bucket = float(item.get("bucket", 0.0))
            if bucket not in bucket_set:
                bucket = min(self.buckets, key=lambda b: abs(b - bucket)) if self.buckets else 0.0
            confidence = float(item.get("confidence", 0.0))
            adv[idx] = bucket
            conf[idx] = confidence
        return adv, conf

    def compute(
        self,
        prompt: str,
        response_text: str,
        sentence_ids: torch.Tensor | None,
        sentence_spans: list[tuple[int, int]] | None,
        sequence_reward: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not bool(getattr(self.cfg, "enable", False)):
            return {"sentence_adv": [], "sentence_conf": [], "debug": {"enabled": False}}

        sentence_texts = []
        if extra and extra.get("sentence_texts") is not None:
            sentence_texts = list(extra["sentence_texts"])
        elif sentence_spans and extra and extra.get("response_tokens") is not None and self.tokenizer is not None:
            token_ids = extra["response_tokens"]
            for start, end in sentence_spans:
                sent_ids = token_ids[start:end]
                sentence_texts.append(self.tokenizer.decode(sent_ids, skip_special_tokens=True))
        else:
            sentence_texts = [s for s in response_text.split("\n") if s.strip()]

        max_chars = int(getattr(self.cfg, "max_chars", 0) or 0)
        if max_chars > 0:
            sentence_texts = [s[:max_chars] for s in sentence_texts]

        max_sentences = int(getattr(self.cfg, "max_sentences", 0) or 0)
        truncated = False
        if max_sentences > 0 and len(sentence_texts) > max_sentences:
            sentence_texts = sentence_texts[:max_sentences]
            truncated = True

        overall_correct = sequence_reward > float(getattr(self.cfg, "correctness_threshold", 0.0))
        rendered_prompt = self._build_prompt(prompt, sentence_texts, overall_correct)
        request = JudgeRequest(
            prompt=prompt,
            response_text=response_text,
            sentences=[{"id": i + 1, "text": s} for i, s in enumerate(sentence_texts)],
            overall_correct=overall_correct,
            buckets=self.buckets,
            extra={**(extra or {}), "rendered_prompt": rendered_prompt},
        )

        def _fallback(reason: str, err: Exception | None = None) -> dict[str, Any]:
            debug = {
                "overall_correct": overall_correct,
                "truncated": truncated,
                "consistency_fixed": False,
                "parse_failed": True,
                "fallback_reason": reason,
            }
            if err is not None:
                debug["parse_error"] = str(err)
            if bool(getattr(self.cfg, "debug_prompt", False)):
                debug["rendered_prompt"] = rendered_prompt
            return {
                "sentence_adv": [0.0] * len(sentence_texts),
                "sentence_conf": [0.0] * len(sentence_texts),
                "debug": debug,
            }

        raw = None
        if extra and "judge_output" in extra:
            raw = extra["judge_output"]
        else:
            if self.judge_client is None:
                raise ValueError("Self judge backend requires precomputed judge_output.")
            raw = self.judge_client.judge(request)

        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="ignore")
            except Exception as exc:
                return _fallback("bytes_decode_failed", exc)
        elif hasattr(raw, "item"):
            try:
                raw = raw.item()
            except Exception:
                pass

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception as exc:
                try:
                    left = raw.find("{")
                    right = raw.rfind("}")
                    if left >= 0 and right > left:
                        raw = json.loads(raw[left : right + 1])
                    else:
                        return _fallback("json_not_found", exc)
                except Exception as exc2:
                    return _fallback("json_decode_failed", exc2)

        if not isinstance(raw, dict):
            return _fallback("json_not_dict")

        try:
            sentence_adv, sentence_conf = self._parse_judge_output(raw, len(sentence_texts))
            sentence_adv = self._apply_confidence(sentence_adv, sentence_conf)
            sentence_adv = self._normalize_sentence_adv(sentence_adv)
            sentence_adv, fixed = self._enforce_consistency(sentence_adv, overall_correct)
        except Exception as exc:
            return _fallback("parse_or_postprocess_failed", exc)

        debug = {
            "overall_correct": overall_correct,
            "truncated": truncated,
            "consistency_fixed": fixed,
            "parse_failed": False,
        }
        if bool(getattr(self.cfg, "debug_prompt", False)):
            debug["rendered_prompt"] = rendered_prompt
        return {"sentence_adv": sentence_adv, "sentence_conf": sentence_conf, "debug": debug}

    def to_token_adv(
        self,
        sentence_adv: list[float],
        sentence_ids: torch.Tensor,
        T: int,
        response_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_adv = torch.zeros((T,), device=sentence_ids.device, dtype=torch.float)
        if sentence_ids is None or sentence_ids.numel() == 0:
            return token_adv
        if response_mask is None:
            response_mask = torch.ones_like(sentence_ids, dtype=torch.float)
        valid_mask = (response_mask > 0) & (sentence_ids >= 0)
        if not torch.any(valid_mask):
            return token_adv

        sids = sentence_ids[valid_mask].detach().cpu().tolist()
        pos = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1).detach().cpu().tolist()
        order: list[int] = []
        seen = set()
        for sid in sids:
            if sid not in seen:
                order.append(sid)
                seen.add(sid)

        sid_to_idx = {sid: i for i, sid in enumerate(order)}
        for p, sid in zip(pos, sids, strict=False):
            idx = sid_to_idx.get(sid, -1)
            if idx < 0 or idx >= len(sentence_adv):
                continue
            token_adv[p] = float(sentence_adv[idx])
        return token_adv

    def compute_batch(
        self,
        prompts: list[str],
        responses: list[str],
        response_tokens: torch.Tensor,
        sentence_ids: torch.Tensor,
        response_mask: torch.Tensor,
        sequence_rewards: torch.Tensor,
        extra_batch: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        bs, seq_len = sentence_ids.shape
        token_adv = torch.zeros((bs, seq_len), device=sentence_ids.device, dtype=torch.float)
        all_adv: list[float] = []
        all_conf: list[float] = []
        pos = neg = zero = 0
        fix_count = 0
        parse_fail_count = 0

        for b in range(bs):
            extra = extra_batch[b] if extra_batch else {}
            sentence_texts, sentence_spans = self._extract_sentence_texts(
                response_tokens[b], sentence_ids[b], response_mask[b]
            )
            extra = {**extra, "sentence_texts": sentence_texts, "response_tokens": response_tokens[b].tolist()}
            try:
                output = self.compute(
                    prompt=prompts[b],
                    response_text=responses[b],
                    sentence_ids=sentence_ids[b],
                    sentence_spans=sentence_spans,
                    sequence_reward=float(sequence_rewards[b].item()),
                    extra=extra,
                )
            except Exception as exc:
                output = {
                    "sentence_adv": [0.0] * len(sentence_texts),
                    "sentence_conf": [0.0] * len(sentence_texts),
                    "debug": {
                        "overall_correct": False,
                        "truncated": False,
                        "consistency_fixed": False,
                        "parse_failed": True,
                        "fallback_reason": "compute_exception",
                        "parse_error": str(exc),
                    },
                }
            sent_adv = output.get("sentence_adv", [])
            sent_conf = output.get("sentence_conf", [])
            all_adv.extend(sent_adv)
            all_conf.extend(sent_conf)
            for v in sent_adv:
                if v > 0:
                    pos += 1
                elif v < 0:
                    neg += 1
                else:
                    zero += 1
            if output.get("debug", {}).get("consistency_fixed", False):
                fix_count += 1
            if output.get("debug", {}).get("parse_failed", False):
                parse_fail_count += 1
            token_adv[b] = self.to_token_adv(sent_adv, sentence_ids[b], seq_len, response_mask=response_mask[b])

        total = max(pos + neg + zero, 1)
        adv_tensor = torch.tensor(all_adv, dtype=torch.float)
        conf_tensor = torch.tensor(all_conf, dtype=torch.float) if all_conf else torch.tensor([0.0])
        metrics = {
            "sentence_judge/adv_mean": float(adv_tensor.mean().item()) if adv_tensor.numel() > 0 else 0.0,
            "sentence_judge/adv_std": float(adv_tensor.std(unbiased=False).item()) if adv_tensor.numel() > 1 else 0.0,
            "sentence_judge/conf_mean": float(conf_tensor.mean().item()) if conf_tensor.numel() > 0 else 0.0,
            "sentence_judge/pos_frac": float(pos / total),
            "sentence_judge/neg_frac": float(neg / total),
            "sentence_judge/zero_frac": float(zero / total),
            "sentence_judge/consistency_fix_rate": float(fix_count / max(bs, 1)),
            "sentence_judge/parse_fail_rate": float(parse_fail_count / max(bs, 1)),
        }
        return token_adv, metrics

    def _extract_sentence_texts(
        self, response_tokens: torch.Tensor, sentence_ids: torch.Tensor, response_mask: torch.Tensor
    ) -> tuple[list[str], list[tuple[int, int]]]:
        if sentence_ids is None or response_tokens is None:
            return [], []
        valid = (response_mask > 0) & (sentence_ids >= 0)
        positions = torch.nonzero(valid, as_tuple=False).squeeze(-1).detach().cpu().tolist()
        if not positions:
            return [], []
        sid_list = sentence_ids[valid].detach().cpu().tolist()
        token_list = response_tokens.detach().cpu().tolist()

        order: list[int] = []
        seen = set()
        sid_to_pos: dict[int, list[int]] = {}
        for sid, pos in zip(sid_list, positions, strict=False):
            if sid not in seen:
                order.append(sid)
                seen.add(sid)
            sid_to_pos.setdefault(sid, []).append(pos)

        sentence_texts: list[str] = []
        sentence_spans: list[tuple[int, int]] = []
        for sid in order:
            pos_list = sid_to_pos.get(sid, [])
            if not pos_list:
                continue
            start = min(pos_list)
            end = max(pos_list) + 1
            sent_tokens = [token_list[i] for i in pos_list]
            if self.tokenizer is not None:
                sentence_texts.append(self.tokenizer.decode(sent_tokens, skip_special_tokens=True))
            else:
                sentence_texts.append(" ".join(str(t) for t in sent_tokens))
            sentence_spans.append((start, end))
        return sentence_texts, sentence_spans
