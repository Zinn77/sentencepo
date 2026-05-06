"""
Extract Phase 1 ep1 val-core mean@1 across 6 math datasets.
Reports per-step trajectory + peak/ep1-end mean6 + per-dataset @ peak.
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict

LOGS = {
    # Phase 1 v1-5 winner (smaller_alpha) ep1
    "v15_winner_qwen3_seed9":   "/root/autodl-tmp/models_v1-5/2026-05-05_phase1_winner_qwen3_math_qwen3_4b_ep1_rand9/verl_v1-5-hidden.log",
    "v15_winner_qwen3_seed37":  "/root/autodl-tmp/models_v1-5/2026-05-05_phase1_winner_qwen3_math_qwen3_4b_ep1_rand37/verl_v1-5-hidden.log",
    # Phase 1 v1-5 alpha_decay ep1
    "v15_decay_qwen3_seed9":    "/root/autodl-tmp/models_v1-5/2026-05-06_phase1_alpha_decay_qwen3_math_qwen3_4b_ep1_rand9/verl_v1-5-hidden.log",
    "v15_decay_qwen3_seed37":   "/root/autodl-tmp/models_v1-5/2026-05-06_phase1_alpha_decay_qwen3_math_qwen3_4b_ep1_rand37/verl_v1-5-hidden.log",
    # Phase 1 Llama
    "v15_winner_llama_seed42":  "/root/autodl-tmp/models_v1-5/2026-05-05_phase1_winner_llama_math_llama3.2-3b_ep1_rand42/verl_v1-5-hidden.log",
    "v15_winner_llama_seed9":   "/root/autodl-tmp/models_v1-5/2026-05-05_phase1_winner_llama_math_llama3.2-3b_ep1_rand9/verl_v1-5-hidden.log",
    # Phase 1 baselines (new ep1 runs)
    "GRPO_qwen3_seed9_NEW":     "/root/autodl-tmp/models_v1-5/grpo_math_qwen3_4b_ep1_rand9/verl_grpo.log",
    "GRPO_qwen3_seed37_NEW":    "/root/autodl-tmp/models_v1-5/grpo_math_qwen3_4b_ep1_rand37/verl_grpo.log",
    # historical baselines (ep3 logs, but same training; truncate at step 58 for ep1 peak)
    "GRPO_qwen3_seed42_ep3":    "/root/autodl-tmp/models_v1-5/grpo_math_qwen3_4b_ep3_rand42/verl_grpo.log",
    "GSPO_qwen3_seed42_ep3":    "/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep3_rand42/verl_gspo.log",
    "GSPO_qwen3_seed37_ep3":    "/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep3_rand37/verl_gspo.log",
    "GSPO_qwen3_seed9_NEW":     "/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep1_rand9/verl_gspo.log",
    # Phase 0 (already have, for sanity)
    "P0_smaller_alpha_seed42":  "/root/autodl-tmp/models_v1-5/2026-05-05_phase0_tune_smaller_alpha_math_qwen3_4b_ep1_rand42/verl_v1-5-hidden.log",
    "P0_alpha_decay_seed42":    "/root/autodl-tmp/models_v1-5/2026-05-05_phase0_tune_alpha_decay_math_qwen3_4b_ep1_rand42/verl_v1-5-hidden.log",
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
    "Hothan/OlympiadBench:OE_MM_maths_en_COMP": "olymp",
}

VAL_RE = re.compile(r"val-core/([^/]+(?:/[^/]+)*?)/reward/mean@1:np\.float64\(([0-9.eE+-]+)\)")
STEP_HEAD_RE = re.compile(r"step:(\d+) - ")


def parse_log(path: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = defaultdict(dict)
    p = Path(path)
    if not p.exists():
        return out
    with open(path) as f:
        for line in f:
            if "val-core/" not in line:
                continue
            ms = STEP_HEAD_RE.search(line)
            if not ms:
                continue
            step = int(ms.group(1))
            for m in VAL_RE.finditer(line):
                out[step][m.group(1)] = float(m.group(2))
    return out


def main():
    print(f"{'tag':<28} {'#steps':>6} {'peak_mean6':>10} {'@step':>6} {'ep1_end':>8}  per-ds @peak (math500 aime24 aime25 amc23 minerva olymp)")
    print("-" * 140)
    summary = {}
    for tag, path in LOGS.items():
        per = parse_log(path)
        if not per:
            print(f"{tag:<28} (missing or no val data)")
            continue
        # Filter steps that have all 6 datasets
        full = {s: d for s, d in per.items() if all(ds in d for ds in DATASETS)}
        if not full:
            # Print partial
            print(f"{tag:<28} {len(per):>6} (partial: only {sum(1 for d in per.values() if all(ds in d for ds in DATASETS))} full blocks)")
            continue
        # ep1 peak: use step <= 55 for fair alignment (ep3 logs have no step 58)
        # ep1 end:  use step == 55 strictly (or largest <= 55 fallback)
        ep1_steps = [s for s in sorted(full) if s <= 55]
        if not ep1_steps:
            continue
        means = [(s, sum(full[s][d] for d in DATASETS)/6.0) for s in ep1_steps]
        peak_step, peak_mean = max(means, key=lambda x: x[1])
        end_step = 55 if 55 in full else max(ep1_steps)
        ep1_end_mean = sum(full[end_step][d] for d in DATASETS) / 6.0
        per_ds = full[peak_step]
        ds_str = " ".join(f"{per_ds[d]:.3f}" for d in DATASETS)
        print(f"{tag:<28} {len(ep1_steps):>6} {peak_mean:>10.4f} {peak_step:>6} {ep1_end_mean:>8.4f}  {ds_str}")
        summary[tag] = (peak_mean, peak_step, ep1_end_mean, per_ds)

    # ====== aggregated tables ======
    print("\n" + "=" * 100)
    print("v1-5 winner (smaller_alpha) Qwen3 multi-seed:")
    seeds = ["P0_smaller_alpha_seed42", "v15_winner_qwen3_seed9", "v15_winner_qwen3_seed37"]
    peaks = [summary[s][0] for s in seeds if s in summary]
    if peaks:
        import statistics as st
        m, sd = st.mean(peaks), st.stdev(peaks) if len(peaks) > 1 else 0
        print(f"  peak mean6 (3 seeds): {peaks}  → mean={m:.4f}  std={sd:.4f}")

    print("\nv1-5 alpha_decay Qwen3 multi-seed:")
    seeds = ["P0_alpha_decay_seed42", "v15_decay_qwen3_seed9", "v15_decay_qwen3_seed37"]
    peaks = [summary[s][0] for s in seeds if s in summary]
    if peaks:
        import statistics as st
        m, sd = st.mean(peaks), st.stdev(peaks) if len(peaks) > 1 else 0
        print(f"  peak mean6 (3 seeds): {peaks}  → mean={m:.4f}  std={sd:.4f}")

    print("\nGRPO Qwen3 multi-seed (ep1 peak):")
    seeds = ["GRPO_qwen3_seed42_ep3", "GRPO_qwen3_seed9_NEW", "GRPO_qwen3_seed37_NEW"]
    peaks = [summary[s][0] for s in seeds if s in summary]
    if peaks:
        import statistics as st
        m, sd = st.mean(peaks), st.stdev(peaks) if len(peaks) > 1 else 0
        print(f"  peak mean6 (3 seeds): {peaks}  → mean={m:.4f}  std={sd:.4f}")

    print("\nGSPO Qwen3 multi-seed (ep1 peak):")
    seeds = ["GSPO_qwen3_seed42_ep3", "GSPO_qwen3_seed9_NEW", "GSPO_qwen3_seed37_ep3"]
    peaks = [summary[s][0] for s in seeds if s in summary]
    ends  = [summary[s][2] for s in seeds if s in summary]
    if peaks:
        import statistics as st
        m, sd = st.mean(peaks), st.stdev(peaks) if len(peaks) > 1 else 0
        em, esd = st.mean(ends), st.stdev(ends) if len(ends) > 1 else 0
        print(f"  peak (3 seeds): {[f'{p:.4f}' for p in peaks]}  → mean={m:.4f}  std={sd:.4f}")
        print(f"  ep1_end (3 seeds): {[f'{e:.4f}' for e in ends]}  → mean={em:.4f}  std={esd:.4f}")

    print("\nv1-5 winner Llama (cross-model):")
    for s in ["v15_winner_llama_seed42", "v15_winner_llama_seed9"]:
        if s in summary:
            print(f"  {s}: peak={summary[s][0]:.4f} @step={summary[s][1]}")


if __name__ == "__main__":
    main()
