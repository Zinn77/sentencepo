"""
Extract val-core mean@1 across 6 datasets from verl logs.
Outputs per-step and per-epoch (step 58 / 116 / 174 = ep1/ep2/ep3) means.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict

LOGS = {
    "M1_hpo_09_C1_L-1": "/root/autodl-tmp/models_v1-5/2026-05-03_day0_M1_hpo_09_C1_ep3_math_qwen3_4b_ep3_rand42/verl_v1-5-hidden.log",
    "M2_hidden_combined_L-18": "/root/autodl-tmp/models_v1-5/2026-05-03_day0_M2_hidden_combined_L-18_ep3_math_qwen3_4b_ep3_rand42/verl_v1-5-hidden.log",
    "M3_GSPO_baseline": "/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep3_rand42/verl_gspo.log",
}

DATASETS = [
    "math500",
    "aime2024",
    "aime2025",
    "amc23",
    "math-ai/minervamath",
    "Hothan/OlympiadBench:OE_MM_maths_en_COMP",
]

EPOCH_STEPS = {"ep1": 58, "ep2": 116, "ep3": 174}

# match e.g. "val-core/math500/reward/mean@1:np.float64(0.678)"
VAL_RE = re.compile(r"val-core/([^/]+(?:/[^/]+)*?)/reward/mean@1:np\.float64\(([0-9.eE+-]+)\)")
# the step is at the *start* of the metrics line, val-core appears later in same line
STEP_HEAD_RE = re.compile(r"step:(\d+) - ")


def parse_log(path: str) -> dict[int, dict[str, float]]:
    """Returns step -> {dataset: mean@1}."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
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
    print(f"{'run':<28} {'step':>4}  " + "  ".join(f"{d.split('/')[-1][:12]:>12}" for d in DATASETS) + f"  {'mean6':>8}")
    rows_per_run: dict[str, list[tuple[int, float, dict[str, float]]]] = {}
    for tag, path in LOGS.items():
        per_step = parse_log(path)
        rows = []
        for step in sorted(per_step):
            scores = per_step[step]
            if len([d for d in DATASETS if d in scores]) < 6:
                continue
            vals = [scores[d] for d in DATASETS]
            mean6 = sum(vals) / 6
            rows.append((step, mean6, scores))
        rows_per_run[tag] = rows
        print(f"--- {tag}  ({len(rows)} val points) ---")
        for step, mean6, scores in rows:
            line = f"{tag:<28} {step:>4}  " + "  ".join(f"{scores[d]:>12.3f}" for d in DATASETS) + f"  {mean6:>8.4f}"
            mark = ""
            for ep_name, ep_step in EPOCH_STEPS.items():
                if step == ep_step:
                    mark = f"  <-- {ep_name}"
            print(line + mark)
        print()

    # per-epoch summary table
    print("\n=== EPOCH SUMMARY (mean@1 over 6 datasets) ===")
    print(f"{'run':<28}  " + "  ".join(f"{ep:>8}" for ep in EPOCH_STEPS) + "  step0(init)")
    init_rows = {tag: rows[0][1] for tag, rows in rows_per_run.items() if rows}
    for tag, rows in rows_per_run.items():
        by_step = {s: m for s, m, _ in rows}
        line = f"{tag:<28}  "
        for ep_name, ep_step in EPOCH_STEPS.items():
            line += f"{by_step.get(ep_step, float('nan')):>8.4f}  "
        line += f"  {init_rows.get(tag, float('nan')):>8.4f}"
        print(line)

    # peak (best) per run for ep1 / cumulative-up-to-ep1, etc.
    print("\n=== PEAK mean6 (max across all val checkpoints up through epoch boundary) ===")
    print(f"{'run':<28}  " + "  ".join(f"{'peak_'+ep:>10}@step" for ep in EPOCH_STEPS))
    for tag, rows in rows_per_run.items():
        line = f"{tag:<28}  "
        for ep_name, ep_step in EPOCH_STEPS.items():
            up_to = [(s, m) for s, m, _ in rows if s <= ep_step]
            if up_to:
                bs, bm = max(up_to, key=lambda x: x[1])
                line += f"{bm:>8.4f}@{bs:<3}  "
            else:
                line += f"{'-':>14}  "
        print(line)


if __name__ == "__main__":
    main()
