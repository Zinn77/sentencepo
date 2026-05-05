"""
Extract val-core mean@1 across 6 math datasets for all 2026-05-03/05 runs.
Reports: per-step, per-epoch (ep1/ep2/ep3 = step 58/116/174), peak overall, peak step.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict

LOGS = {
    "M1_L-1_default":          "/root/autodl-tmp/models_v1-5/2026-05-03_day0_M1_hpo_09_C1_ep3_math_qwen3_4b_ep3_rand42/verl_v1-5-hidden.log",
    "M2_L-18_default":         "/root/autodl-tmp/models_v1-5/2026-05-03_day0_M2_hidden_combined_L-18_ep3_math_qwen3_4b_ep3_rand42/verl_v1-5-hidden.log",
    "M2_L-18_slpa33_scr33":    "/root/autodl-tmp/models_v1-5/2026-05-03_day0_M2_hidden_combined_L-18_ep3_slpa33_scr33_math_qwen3_4b_ep3_rand42/verl_v1-5-hidden.log",
    "M2_L-18_slpa55_scr55":    "/root/autodl-tmp/models_v1-5/2026-05-03_day0_M2_hidden_combined_L-18_ep3_slpa55_scr55_math_qwen3_4b_ep3_rand42/verl_v1-5-hidden.log",
    "GRPO_qwen3_seed42":       "/root/autodl-tmp/models_v1-5/grpo_math_qwen3_4b_ep3_rand42/verl_grpo.log",
    "GSPO_qwen3_seed42":       "/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep3_rand42/verl_gspo.log",
    "GSPO_qwen3_seed37":       "/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep3_rand37/verl_gspo.log",
}

DATASETS = [
    "math500",
    "aime2024",
    "aime2025",
    "amc23",
    "math-ai/minervamath",
    "Hothan/OlympiadBench:OE_MM_maths_en_COMP",
]

DS_SHORT = {
    "math500": "math500",
    "aime2024": "aime24",
    "aime2025": "aime25",
    "amc23": "amc23",
    "math-ai/minervamath": "minerva",
    "Hothan/OlympiadBench:OE_MM_maths_en_COMP": "olympiad",
}

EPOCH_STEPS = {"ep1": 58, "ep2": 116, "ep3": 174}

VAL_RE = re.compile(r"val-core/([^/]+(?:/[^/]+)*?)/reward/mean@1:np\.float64\(([0-9.eE+-]+)\)")
STEP_HEAD_RE = re.compile(r"step:(\d+) - ")


def parse_log(path: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = defaultdict(dict)
    p = Path(path)
    if not p.exists():
        print(f"!! missing: {path}", file=sys.stderr)
        return out
    with open(path) as f:
        for line in f:
            if "val-core/" not in line:
                continue
            m_step = STEP_HEAD_RE.search(line)
            if not m_step:
                continue
            step = int(m_step.group(1))
            for m in VAL_RE.finditer(line):
                ds = m.group(1)
                val = float(m.group(2))
                out[step][ds] = val
    return out


def main():
    rows_per_run: dict[str, list[tuple[int, float, dict[str, float]]]] = {}
    for tag, path in LOGS.items():
        per_step = parse_log(path)
        rows = []
        for step in sorted(per_step):
            scores = per_step[step]
            if any(d not in scores for d in DATASETS):
                continue
            mean6 = sum(scores[d] for d in DATASETS) / 6.0
            rows.append((step, mean6, scores))
        rows_per_run[tag] = rows

    # ===== per-step trajectory =====
    print("=" * 100)
    print("PER-STEP mean@1 (6 datasets)")
    print("=" * 100)
    head = f"{'run':<28} {'step':>5}  " + " ".join(f"{DS_SHORT[d]:>8}" for d in DATASETS) + f"  {'mean6':>8}  mark"
    for tag, rows in rows_per_run.items():
        print("-" * len(head))
        print(head)
        for step, mean6, sc in rows:
            mark = ""
            for ep, st in EPOCH_STEPS.items():
                if step == st:
                    mark = ep
            line = f"{tag:<28} {step:>5}  " + " ".join(f"{sc[d]:>8.3f}" for d in DATASETS) + f"  {mean6:>8.4f}  {mark}"
            print(line)

    # ===== epoch boundary =====
    print()
    print("=" * 100)
    print("EPOCH-END mean6 (val at ep boundary, exact step)")
    print("=" * 100)
    print(f"{'run':<28}  " + "  ".join(f"{ep:>10}" for ep in EPOCH_STEPS))
    for tag, rows in rows_per_run.items():
        by = {s: m for s, m, _ in rows}
        line = f"{tag:<28}  " + "  ".join(f"{by.get(EPOCH_STEPS[ep], float('nan')):>10.4f}" for ep in EPOCH_STEPS)
        print(line)

    # ===== peak overall =====
    print()
    print("=" * 100)
    print("PEAK mean6 (across all val checkpoints, all epochs)")
    print("=" * 100)
    print(f"{'run':<28}  {'peak':>8}  {'peak_step':>10}  {'last_step':>10}  {'last':>8}")
    for tag, rows in rows_per_run.items():
        if not rows:
            continue
        bs, bm = max(((s, m) for s, m, _ in rows), key=lambda x: x[1])
        ls, lm, _ = rows[-1]
        print(f"{tag:<28}  {bm:>8.4f}  {bs:>10}  {ls:>10}  {lm:>8.4f}")

    # ===== peak per-dataset (per-ds peak across epochs, then mean) =====
    print()
    print("=" * 100)
    print("PEAK PER-DATASET (each dataset's own peak, then mean)")
    print("=" * 100)
    print(f"{'run':<28}  " + " ".join(f"{DS_SHORT[d]:>8}" for d in DATASETS) + f"  {'peak_avg':>8}")
    for tag, rows in rows_per_run.items():
        if not rows:
            continue
        per_ds_peaks = {d: max(sc[d] for _, _, sc in rows) for d in DATASETS}
        avg = sum(per_ds_peaks[d] for d in DATASETS) / 6.0
        print(f"{tag:<28}  " + " ".join(f"{per_ds_peaks[d]:>8.3f}" for d in DATASETS) + f"  {avg:>8.4f}")

    # ===== peak per-dataset detail (which step) =====
    print()
    print("=" * 100)
    print("PER-DATASET PEAK STEP (step where each dataset peaked)")
    print("=" * 100)
    print(f"{'run':<28}  " + " ".join(f"{DS_SHORT[d]:>8}" for d in DATASETS))
    for tag, rows in rows_per_run.items():
        if not rows:
            continue
        cells = []
        for d in DATASETS:
            best_step, best_val = max(((s, sc[d]) for s, _, sc in rows), key=lambda x: x[1])
            cells.append(f"{best_step:>8}")
        print(f"{tag:<28}  " + " ".join(cells))


if __name__ == "__main__":
    main()
