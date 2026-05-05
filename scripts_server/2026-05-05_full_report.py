"""
Generate the full advisor-report data tables (ep1 / ep3 / peak per-dataset).

Reads `~/autodl-tmp/models_v1-5/<exp_dir>/verl_*.log` for all known runs and
prints comparison tables. Used to refresh `CCdocs/2026-05-05_advisor_report.md`.

Run on the host where logs are present (autodl GPU box that has the synced
verl_*.log files).
"""
from __future__ import annotations
import re
from pathlib import Path

DATASETS = [
    "math500",
    "aime2024",
    "aime2025",
    "amc23",
    "math-ai/minervamath",
    "Hothan/OlympiadBench:OE_MM_maths_en_COMP",
]
SHORT = {
    "math500": "math500",
    "aime2024": "aime24",
    "aime2025": "aime25",
    "amc23": "amc23",
    "math-ai/minervamath": "minerva",
    "Hothan/OlympiadBench:OE_MM_maths_en_COMP": "olympiad",
}
LOGS = {
    "M1: hpo_09_C1 L=-1 (Bug-G不影响) seed=42":
        "2026-05-03_day0_M1_hpo_09_C1_ep3_math_qwen3_4b_ep3_rand42",
    "M2_default: combined L=-18 (slpa0.05/scr0.02) seed=42":
        "2026-05-03_day0_M2_hidden_combined_L-18_ep3_math_qwen3_4b_ep3_rand42",
    "M2_slpa33: combined L=-18 (slpa0.03/scr0.03) seed=42":
        "2026-05-03_day0_M2_hidden_combined_L-18_ep3_slpa33_scr33_math_qwen3_4b_ep3_rand42",
    "M2_slpa55: combined L=-18 (slpa0.05/scr0.05) seed=42":
        "2026-05-03_day0_M2_hidden_combined_L-18_ep3_slpa55_scr55_math_qwen3_4b_ep3_rand42",
    "GRPO Qwen3 seed=42": "grpo_math_qwen3_4b_ep3_rand42",
    "GSPO Qwen3 seed=42": "gspo_math_qwen3_4b_ep3_rand42",
    "GSPO Qwen3 seed=37": "gspo_math_qwen3_4b_ep3_rand37",
    "GRPO Llama seed=42 (collapsed)": "grpo_math_llama3.2-3b_ep3_rand42",
    "GSPO Llama seed=42": "gspo_math_llama3.2-3b_ep3_rand42",
}
ROOT = Path.home() / "autodl-tmp" / "models_v1-5"

VAL_RE = re.compile(r"val-core/([^/]+(?:/[^/]+)*?)/reward/mean@1:np\.float64\(([0-9.eE+-]+)\)")
HEAD_RE = re.compile(r"step:(\d+) - ")


def parse(p: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    fp = list(p.glob("verl_*.log"))
    if not fp:
        return out
    with open(fp[0]) as f:
        for line in f:
            if "val-core/" not in line:
                continue
            m = HEAD_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            scores = {}
            for mm in VAL_RE.finditer(line):
                scores[mm.group(1)] = float(mm.group(2))
            if all(d in scores for d in DATASETS):
                out[step] = scores
    return out


def m6(scs: dict[str, float]) -> float:
    return sum(scs[d] for d in DATASETS) / 6


def main():
    data = {n: parse(ROOT / d) for n, d in LOGS.items()}

    print(f"=== ep1 (step 55) / ep3 (step 174) / peak summary ===")
    print(f"{'run':<58} {'ep1@55':>8} {'ep3@174':>9} {'peak':>7} {'@step':>6} {'lastStep':>9}")
    for n, rows in data.items():
        if not rows:
            print(f"{n:<58} no data")
            continue
        ep1 = m6(rows[55]) if 55 in rows else float('nan')
        ep3 = m6(rows[174]) if 174 in rows else float('nan')
        bs = max(rows, key=lambda s: m6(rows[s]))
        bv = m6(rows[bs])
        last = max(rows)
        ep3_str = f"{ep3:>9.4f}" if ep3 == ep3 else f"{'⏳':>9}"
        ep1_str = f"{ep1:>8.4f}" if ep1 == ep1 else f"{'-':>8}"
        print(f"{n:<58} {ep1_str} {ep3_str} {bv:>7.4f} {bs:>6d} {last:>9d}")

    for label, target in [("ep1 (step 55)", 55), ("ep3 (step 174)", 174)]:
        print(f"\n=== {label} per-dataset ===")
        print(f"{'run':<58} " + "  ".join(f"{SHORT[d]:>9}" for d in DATASETS))
        for n, rows in data.items():
            if target not in rows:
                continue
            print(f"{n:<58} " + "  ".join(f"{rows[target][d]:>9.4f}" for d in DATASETS))

    print("\n=== peak (across all val checkpoints, per-dataset) ===")
    print(f"{'run':<58} " + "  ".join(f"{SHORT[d]:>9}" for d in DATASETS))
    for n, rows in data.items():
        if not rows:
            continue
        pk = {d: max(rows[s][d] for s in rows) for d in DATASETS}
        print(f"{n:<58} " + "  ".join(f"{pk[d]:>9.4f}" for d in DATASETS))

    print("\n=== Best-of comparison: v1-5 (M1+M2 trio) vs Qwen3 baseline ===")
    v15 = [
        "M1: hpo_09_C1 L=-1 (Bug-G不影响) seed=42",
        "M2_default: combined L=-18 (slpa0.05/scr0.02) seed=42",
        "M2_slpa33: combined L=-18 (slpa0.03/scr0.03) seed=42",
        "M2_slpa55: combined L=-18 (slpa0.05/scr0.05) seed=42",
    ]
    ctrls = ["GRPO Qwen3 seed=42", "GSPO Qwen3 seed=42", "GSPO Qwen3 seed=37"]
    for label, target in [("ep1@55", 55), ("ep3@174", 174)]:
        v15_v = [m6(data[n][target]) for n in v15 if target in data.get(n, {})]
        ctrl_v = [m6(data[n][target]) for n in ctrls if target in data.get(n, {})]
        if not v15_v or not ctrl_v:
            print(f"  {label}: insufficient data")
            continue
        print(f"  {label:<8}: best v1-5={max(v15_v):.4f}, best ctrl={max(ctrl_v):.4f}, diff={max(v15_v)-max(ctrl_v):+.4f}")

    v15_peaks = [max(m6(data[n][s]) for s in data[n]) for n in v15 if data[n]]
    ctrl_peaks = [max(m6(data[n][s]) for s in data[n]) for n in ctrls if data[n]]
    if v15_peaks and ctrl_peaks:
        print(f"  peak    : best v1-5={max(v15_peaks):.4f}, best ctrl={max(ctrl_peaks):.4f}, "
              f"diff={max(v15_peaks)-max(ctrl_peaks):+.4f}")


if __name__ == "__main__":
    main()
