#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project root is importable when running as a script.
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def build_prompt(prompt: str, sentence_texts: list[str], overall_correct: bool, buckets: list[float]) -> str:
    bucket_str = ", ".join(str(b) for b in buckets)
    numbered = "\n".join(f"S{i + 1}: {s}" for i, s in enumerate(sentence_texts))
    return (
        "You are a sentence-level judge. "
        "Score each sentence with a discrete advantage bucket and a confidence in [0,1].\n"
        f"Allowed buckets: [{bucket_str}].\n"
        f"overall_correct = {str(bool(overall_correct)).lower()}\n\n"
        "PROMPT:\n"
        f"{prompt}\n\n"
        "RESPONSE (sentence numbered):\n"
        f"{numbered}\n\n"
        "Output strict JSON only:\n"
        '{"sentences":[{"id":1,"bucket":0.25,"confidence":0.8,"reason":"..."}]}'
    )


def split_sentences(tokenizer, text: str, max_sentences: int, max_chars: int) -> list[str]:
    """Split text into sentences using the same token-level logic as RL training."""
    from verl.utils.sentence_utils import split_text_to_sentences

    return split_text_to_sentences(
        tokenizer, text, min_sent_tokens=6, max_sentences=max_sentences, max_chars=max_chars
    )


def read_pre_split_sentences(row: dict[str, Any], max_sentences: int, max_chars: int) -> list[str]:
    raw = row.get("output_sentences", None)
    if not isinstance(raw, list):
        return []

    sentence_texts: list[str] = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        if max_chars > 0:
            s = s[:max_chars]
        sentence_texts.append(s)
        if max_sentences > 0 and len(sentence_texts) >= max_sentences:
            break
    return sentence_texts


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        text = resp.read().decode("utf-8")
    return json.loads(text)


def extract_json_obj(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        return json.loads(text[left : right + 1])
    raise ValueError("No JSON object found in teacher output")


def normalize_teacher_json(raw: dict[str, Any], n_sentences: int, buckets: list[float]) -> dict[str, Any]:
    '''
    1. 先尝试从返回文本提取 JSON 对象（提取失败会重试，最终失败则该条跳过）。
    2. 对 sentences 列表逐条规范化：
    - id 必须是整数且在 1 到 n_sentences 范围内，否则丢弃。
    - bucket 会吸附到允许桶值中最近的一个。
    - confidence 会被裁剪到 0 到 1。
    - reason 会截断。
    3. 对缺失句子自动补默认项，reason 为 missing_from_teacher。
    '''
    allowed = list(buckets)
    by_id: dict[int, dict[str, Any]] = {}
    for item in raw.get("sentences", []):
        try:
            sid = int(item.get("id", -1))
        except Exception:
            continue
        if sid < 1 or sid > n_sentences:
            continue

        bucket = float(item.get("bucket", 0.0))
        if allowed:
            bucket = min(allowed, key=lambda b: abs(b - bucket))
        conf = float(item.get("confidence", 0.0))
        conf = min(max(conf, 0.0), 1.0)
        by_id[sid] = {
            "id": sid,
            "bucket": bucket,
            "confidence": conf,
            "reason": str(item.get("reason", "distilled"))[:256],
        }

    normalized = []
    default_bucket = 0.0 if 0.0 in allowed else (allowed[len(allowed) // 2] if allowed else 0.0)
    for sid in range(1, n_sentences + 1):
        normalized.append(
            by_id.get(
                sid,
                {"id": sid, "bucket": default_bucket, "confidence": 0.0, "reason": "missing_from_teacher"},
            )
        )
    return {"sentences": normalized}


def call_teacher_judge(
    base_url: str,
    model: str,
    judge_prompt: str,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    api_key: str = "",
) -> tuple[dict[str, Any], str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": judge_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    err: Exception | None = None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    for i in range(retries + 1):
        try:
            resp = post_json(
                f"{base_url.rstrip('/')}/chat/completions",
                body,
                timeout_s=timeout_s,
                headers=headers,
            )
            content = str(resp["choices"][0]["message"]["content"])
            return extract_json_obj(content), content
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            err = exc
            time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"Teacher judge failed after retries: {err}")


def read_jsonl_files(glob_pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(glob_pattern)):
        p = Path(path)
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SJA distillation SFT parquet from rollout jsonl.")
    parser.add_argument("--input-glob", type=str, required=True, help="Glob, e.g. /path/rollout_debug/*.jsonl")
    parser.add_argument("--out-train", type=str, default="", help="Output train parquet (used with internal split).")
    parser.add_argument("--out-val", type=str, default="", help="Output val parquet (used with internal split).")
    parser.add_argument(
        "--out-parquet",
        type=str,
        default="",
        help="Single output parquet (no internal train/val split). "
        "Use this when input is already split by prepare_judge_curriculum_data.py.",
    )
    parser.add_argument("--teacher-base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument(
        "--teacher-api-key",
        type=str,
        default="",
        help="Optional OpenAI-compatible API key. If empty, read SJA_TEACHER_API_KEY/DASHSCOPE_API_KEY/OPENAI_API_KEY.",
    )
    parser.add_argument("--teacher-max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sentences", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=384)
    parser.add_argument("--buckets", type=str, default="0.75,0.25,0.0,-0.25,-0.75")
    parser.add_argument(
        "--overall-correct-source",
        type=str,
        choices=["score", "zero"],
        default="score",
        help="score: use row['score'] > 0; zero: always false",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="HuggingFace tokenizer name or path. Required for token-level sentence splitting.",
    )
    parser.add_argument(
        "--min-sent-tokens",
        type=int,
        default=6,
        help="Minimum tokens per sentence; short sentences are merged (must match RL training config).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Print progress every N rows (0 to disable periodic logs).",
    )
    parser.add_argument(
        "--progress-log-file",
        type=str,
        default="",
        help="Optional path to save progress logs. If empty and --out-parquet is set, defaults to <out-parquet>.progress.log.",
    )
    parser.add_argument(
        "--records-log-file",
        type=str,
        default="",
        help="Incremental distilled records jsonl path. Each successful sample is appended immediately. "
        "If the file exists, script resumes from existing records.",
    )
    parser.add_argument(
        "--raw-log-file",
        type=str,
        default="",
        help="Optional teacher raw-response jsonl path for debugging. "
        "Each attempted API sample appends one event (ok/error) including raw text when available.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for teacher API calls (1 = sequential).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    progress_log_file = args.progress_log_file.strip()
    if not progress_log_file and args.out_parquet:
        progress_log_file = f"{args.out_parquet}.progress.log"

    records_log_file = args.records_log_file.strip()
    if not records_log_file and args.out_parquet:
        records_log_file = f"{args.out_parquet}.records.jsonl"

    raw_log_file = args.raw_log_file.strip()
    if not raw_log_file and args.out_parquet:
        raw_log_file = f"{args.out_parquet}.teacher_raw.jsonl"

    num_workers = max(1, int(args.num_workers))

    def log_progress(message: str) -> None:
        print(message, flush=True)
        if progress_log_file:
            p = Path(progress_log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(message + "\n")

    if progress_log_file:
        p = Path(progress_log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write("")
        log_progress(f"Progress log: {progress_log_file}")

    if raw_log_file:
        rp = Path(raw_log_file)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", encoding="utf-8") as f:
            f.write("")
        log_progress(f"Raw teacher log: {raw_log_file}")

    # Load tokenizer for token-level sentence splitting (consistent with RL training).
    if not args.tokenizer:
        raise RuntimeError(
            "--tokenizer is required (e.g. Qwen/Qwen3-4B-Instruct-2507). "
            "This ensures sentence boundaries match RL training."
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"Loaded tokenizer: {args.tokenizer}")

    teacher_api_key = (
        args.teacher_api_key
        or os.environ.get("SJA_TEACHER_API_KEY", "")
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )

    if "dashscope.aliyuncs.com" in args.teacher_base_url and not teacher_api_key:
        raise RuntimeError(
            "DashScope endpoint requires API key. Set --teacher-api-key or DASHSCOPE_API_KEY."
        )

    buckets = [float(x.strip()) for x in args.buckets.split(",") if x.strip()]
    rows = read_jsonl_files(args.input_glob)
    random.shuffle(rows)
    max_samples = int(args.max_samples)
    sample_limit = max_samples if max_samples > 0 else None
    total_planned = min(len(rows), sample_limit) if sample_limit is not None else len(rows)
    log_progress(f"Loaded input rows: {len(rows)}, planned rows: {total_planned}")

    records: list[dict[str, str]] = []
    done_row_uids: set[str] = set()
    legacy_resume_count = 0
    if records_log_file and Path(records_log_file).exists():
        with Path(records_log_file).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and "prompt" in rec and "response" in rec:
                        loaded = {
                            "prompt": str(rec["prompt"]),
                            "response": str(rec["response"]),
                        }
                        row_uid = rec.get("row_uid", None)
                        if isinstance(row_uid, str) and row_uid:
                            loaded["row_uid"] = row_uid
                            done_row_uids.add(row_uid)
                        else:
                            legacy_resume_count += 1
                        records.append(loaded)
                except json.JSONDecodeError:
                    continue
        log_progress(
            f"Resume from records: {records_log_file}, existing={len(records)}, "
            f"uid_tracked={len(done_row_uids)}, legacy={legacy_resume_count}"
        )
    elif records_log_file:
        rp = Path(records_log_file)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", encoding="utf-8") as f:
            f.write("")
        log_progress(f"Records log: {records_log_file}")

    seen = len(records)
    skipped = 0
    processed = 0
    api_done = 0
    t0 = time.time()
    log_every = max(0, int(args.log_every))

    def emit_progress(force: bool = False) -> None:
        if total_planned <= 0:
            return
        if (not force) and log_every <= 0:
            return
        if (not force) and (processed != 1) and (processed % log_every != 0):
            return
        elapsed = max(1e-6, time.time() - t0)
        speed = processed / elapsed
        remaining = max(0, total_planned - processed)
        eta_sec = int(remaining / max(speed, 1e-6))
        log_progress(
            f"Progress: {processed}/{total_planned} ({processed / max(total_planned, 1):.1%}), "
            f"distilled={seen}, skipped={skipped}, api_done={api_done}, speed={speed:.2f} rows/s, eta={eta_sec}s"
        )

    def build_row_uid(row: dict[str, Any]) -> str:
        payload = {
            "input": str(row.get("input", "")).strip(),
            "output": str(row.get("output", "")).strip(),
            "score": row.get("score", None),
            "step": row.get("step", None),
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def persist_record(rec: dict[str, str]) -> None:
        records.append(rec)
        if records_log_file:
            with Path(records_log_file).open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def persist_raw_event(event: dict[str, Any]) -> None:
        if not raw_log_file:
            return
        with Path(raw_log_file).open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def prepare_job(row: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal skipped
        prompt = str(row.get("input", "")).strip()
        response = str(row.get("output", "")).strip()
        if not prompt or not response:
            skipped += 1
            return None

        row_uid = build_row_uid(row)
        if row_uid in done_row_uids:
            return None

        sentence_texts = read_pre_split_sentences(row, max_sentences=args.max_sentences, max_chars=args.max_chars)
        if not sentence_texts:
            sentence_texts = split_sentences(
                tokenizer,
                response,
                max_sentences=args.max_sentences,
                max_chars=args.max_chars,
            )
        if not sentence_texts:
            skipped += 1
            return None

        if args.overall_correct_source == "score":
            try:
                overall_correct = float(row.get("score", 0.0)) > 0.0
            except Exception:
                overall_correct = False
        else:
            overall_correct = False

        return {
            "row_uid": row_uid,
            "judge_prompt": build_prompt(prompt, sentence_texts, overall_correct=overall_correct, buckets=buckets),
            "n_sentences": len(sentence_texts),
        }

    def run_api_job(job: dict[str, Any]) -> dict[str, Any]:
        raw, teacher_raw_content = call_teacher_judge(
            base_url=args.teacher_base_url,
            model=args.teacher_model,
            judge_prompt=job["judge_prompt"],
            max_tokens=args.teacher_max_tokens,
            timeout_s=args.timeout_s,
            retries=args.retries,
            api_key=teacher_api_key,
        )
        normalized = normalize_teacher_json(raw, n_sentences=int(job["n_sentences"]), buckets=buckets)
        return {
            "row_uid": str(job["row_uid"]),
            "prompt": str(job["judge_prompt"]),
            "response": json.dumps(normalized, ensure_ascii=False),
            "teacher_raw_content": teacher_raw_content,
            "teacher_raw_json": json.dumps(raw, ensure_ascii=False),
            "n_sentences": int(job["n_sentences"]),
        }

    if num_workers == 1:
        for row in rows:
            if sample_limit is not None and seen >= sample_limit:
                break

            processed += 1
            emit_progress()

            if processed <= legacy_resume_count:
                continue

            job = prepare_job(row)
            if job is None:
                continue

            try:
                rec = run_api_job(job)
            except Exception as exc:
                persist_raw_event(
                    {
                        "status": "error",
                        "row_uid": str(job["row_uid"]),
                        "n_sentences": int(job["n_sentences"]),
                        "error": str(exc),
                    }
                )
                skipped += 1
                continue

            persist_record(rec)
            persist_raw_event(
                {
                    "status": "ok",
                    "row_uid": rec["row_uid"],
                    "n_sentences": int(rec.get("n_sentences", 0)),
                    "teacher_raw_content": str(rec.get("teacher_raw_content", "")),
                    "teacher_raw_json": str(rec.get("teacher_raw_json", "")),
                }
            )
            done_row_uids.add(rec["row_uid"])
            seen += 1
            api_done += 1
    else:
        log_progress(f"Using multithreaded API mode: num_workers={num_workers}")
        inflight: dict[concurrent.futures.Future, dict[str, Any]] = {}
        rows_iter = iter(rows)
        exhausted = False
        max_inflight = max(num_workers * 2, 4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            while True:
                while (not exhausted) and len(inflight) < max_inflight:
                    if sample_limit is not None and seen >= sample_limit:
                        exhausted = True
                        break
                    try:
                        row = next(rows_iter)
                    except StopIteration:
                        exhausted = True
                        break

                    processed += 1
                    emit_progress()

                    if processed <= legacy_resume_count:
                        continue

                    job = prepare_job(row)
                    if job is None:
                        continue

                    fut = executor.submit(run_api_job, job)
                    inflight[fut] = job

                if not inflight:
                    if exhausted:
                        break
                    continue

                done, _ = concurrent.futures.wait(
                    inflight.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    job = inflight.pop(fut, None)
                    try:
                        rec = fut.result()
                    except Exception as exc:
                        if job is not None:
                            persist_raw_event(
                                {
                                    "status": "error",
                                    "row_uid": str(job["row_uid"]),
                                    "n_sentences": int(job["n_sentences"]),
                                    "error": str(exc),
                                }
                            )
                        skipped += 1
                        continue

                    persist_record(rec)
                    persist_raw_event(
                        {
                            "status": "ok",
                            "row_uid": rec["row_uid"],
                            "n_sentences": int(rec.get("n_sentences", 0)),
                            "teacher_raw_content": str(rec.get("teacher_raw_content", "")),
                            "teacher_raw_json": str(rec.get("teacher_raw_json", "")),
                        }
                    )
                    done_row_uids.add(rec["row_uid"])
                    seen += 1
                    api_done += 1

                    if log_every > 0 and api_done % log_every == 0:
                        emit_progress(force=True)

    emit_progress(force=True)

    if not records:
        raise RuntimeError("No distilled samples produced. Check teacher server/model and input jsonl quality.")

    output_info: dict[str, Any] = {
        "total_input_rows": len(rows),
        "distilled_rows": len(records),
        "skipped_rows": skipped,
        "teacher_auth_enabled": bool(teacher_api_key),
    }
    export_records = [{"prompt": r["prompt"], "response": r["response"]} for r in records]

    if args.out_parquet:
        # Single-file output: input is already split externally (e.g. by prepare_judge_curriculum_data.py).
        out_path = Path(args.out_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(export_records).to_parquet(out_path, index=False)
        output_info["out_parquet"] = str(out_path)
        output_info["rows"] = len(export_records)
    elif args.out_train and args.out_val:
        # Internal train/val split for simple usage without prepare_judge_curriculum_data.py.
        val_size = max(1, int(len(export_records) * args.val_ratio))
        train_records = export_records[val_size:]
        val_records = export_records[:val_size]

        out_train = Path(args.out_train)
        out_val = Path(args.out_val)
        out_train.parent.mkdir(parents=True, exist_ok=True)
        out_val.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(train_records).to_parquet(out_train, index=False)
        pd.DataFrame(val_records).to_parquet(out_val, index=False)
        output_info.update(
            {"out_train": str(out_train), "out_val": str(out_val),
             "train_rows": len(train_records), "val_rows": len(val_records)}
        )
    else:
        raise RuntimeError("Specify either --out-parquet or both --out-train and --out-val.")

    print(json.dumps(output_info, ensure_ascii=False))
    log_progress(json.dumps(output_info, ensure_ascii=False))


if __name__ == "__main__":
    main()
