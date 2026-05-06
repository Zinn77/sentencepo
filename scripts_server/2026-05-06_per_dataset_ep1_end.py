"""Per-dataset ep1_end mean (3 seeds) for v1-5 alpha_decay vs GSPO/GRPO/v1-5 winner.

Question: 去掉 1-2 个数据集后，v1-5 alpha_decay 能否稳稳赢 baseline？
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict
import statistics as st

LOGS = {
    # 3 seeds per method
    "v15_winner_qwen3":  ["/root/autodl-tmp/models_v1-5/2026-05-05_phase0_tune_smaller_alpha_math_qwen3_4b_ep1_rand42/verl_v1-5-hidden.log",
                          "/root/autodl-tmp/models_v1-5/2026-05-05_phase1_winner_qwen3_math_qwen3_4b_ep1_rand9/verl_v1-5-hidden.log",
                          "/root/autodl-tmp/models_v1-5/2026-05-05_phase1_winner_qwen3_math_qwen3_4b_ep1_rand37/verl_v1-5-hidden.log"],
    "v15_alpha_decay":   ["/root/autodl-tmp/models_v1-5/2026-05-05_phase0_tune_alpha_decay_math_qwen3_4b_ep1_rand42/verl_v1-5-hidden.log",
                          "/root/autodl-tmp/models_v1-5/2026-05-06_phase1_alpha_decay_qwen3_math_qwen3_4b_ep1_rand9/verl_v1-5-hidden.log",
                          "/root/autodl-tmp/models_v1-5/2026-05-06_phase1_alpha_decay_qwen3_math_qwen3_4b_ep1_rand37/verl_v1-5-hidden.log"],
    "GRPO_qwen3":        ["/root/autodl-tmp/models_v1-5/grpo_math_qwen3_4b_ep3_rand42/verl_grpo.log",
                          "/root/autodl-tmp/models_v1-5/grpo_math_qwen3_4b_ep1_rand9/verl_grpo.log",
                          "/root/autodl-tmp/models_v1-5/grpo_math_qwen3_4b_ep1_rand37/verl_grpo.log"],
    "GSPO_qwen3":        ["/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep3_rand42/verl_gspo.log",
                          "/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep1_rand9/verl_gspo.log",
                          "/root/autodl-tmp/models_v1-5/gspo_math_qwen3_4b_ep3_rand37/verl_gspo.log"],
}

DATASETS = ["math500", "aime2024", "aime2025", "amc23", "math-ai/minervamath", "Hothan/OlympiadBench:OE_MM_maths_en_COMP"]
DS_SHORT = {DATASETS[0]:"math500", DATASETS[1]:"aime24", DATASETS[2]:"aime25", DATASETS[3]:"amc23", DATASETS[4]:"minerva", DATASETS[5]:"olymp"}

VAL_RE = re.compile(r"val-core/([^/]+(?:/[^/]+)*?)/reward/mean@1:np\.float64\(([0-9.eE+-]+)\)")
STEP_HEAD_RE = re.compile(r"step:(\d+) - ")


def parse(path: str) -> dict[int, dict[str, float]]:
    out = defaultdict(dict)
    if not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            if "val-core/" not in line: continue
            ms = STEP_HEAD_RE.search(line)
            if not ms: continue
            s = int(ms.group(1))
            for m in VAL_RE.finditer(line):
                out[s][m.group(1)] = float(m.group(2))
    return out


def ep1_end_dict(per_step, strict_step=55):
    """Return the per-dataset dict at strict_step (default 55) for fair alignment.

    Reason: ep1 native runs have val at step 58 (epoch-end forced val), while ep3
    runs only have val at step 0,5,...,55 within ep1 window. Using step 55 across
    all methods removes the 3-step training bias (~5% epoch progress)."""
    full = {s:d for s,d in per_step.items() if all(ds in d for ds in DATASETS)}
    if strict_step in full:
        return strict_step, full[strict_step]
    # fallback: largest step <= strict_step
    candidates = [s for s in full if s <= strict_step]
    if not candidates: return None
    s_end = max(candidates)
    return s_end, full[s_end]


def main():
    # method -> {dataset_short: [v_seed1, v_seed2, v_seed3]}
    by_method = {}
    for tag, logs in LOGS.items():
        by_method[tag] = defaultdict(list)
        for log in logs:
            ps = parse(log)
            r = ep1_end_dict(ps)
            if r is None:
                print(f"!! missing ep1_end for {tag} :: {log}")
                continue
            s_end, d = r
            for ds in DATASETS:
                by_method[tag][DS_SHORT[ds]].append(d[ds])

    # ===== Print per-dataset 3-seed mean ± std =====
    methods = ["GSPO_qwen3", "GRPO_qwen3", "v15_winner_qwen3", "v15_alpha_decay"]
    short_ds = [DS_SHORT[d] for d in DATASETS]

    print("=" * 110)
    print("Per-dataset ep1_end mean@1 (3-seed mean ± std)")
    print("=" * 110)
    head = f"{'method':<22}" + "".join(f"{s:>14}" for s in short_ds) + f"{'mean6':>10}" + f"{'mean5_no_olymp':>16}"
    print(head)
    print("-" * 110)
    for m in methods:
        line = f"{m:<22}"
        means = {}
        for s in short_ds:
            vals = by_method[m][s]
            mean = st.mean(vals) if vals else 0.0
            std  = st.stdev(vals) if len(vals) > 1 else 0.0
            means[s] = mean
            line += f"  {mean:.3f}±{std:.3f}"
        mean6 = sum(means.values()) / 6
        mean5_no_olymp = (mean6*6 - means['olymp']) / 5
        line += f"  {mean6:>8.4f}" + f"  {mean5_no_olymp:>14.4f}"
        print(line)

    # ===== Pairwise gap: v1-5 alpha_decay vs each baseline =====
    print("\n" + "=" * 110)
    print("Δ(v15_alpha_decay - baseline) per dataset (positive = v1-5 wins)")
    print("=" * 110)
    print(f"{'baseline':<12}" + "".join(f"{s:>10}" for s in short_ds))
    print("-" * 110)
    for base in ["GSPO_qwen3", "GRPO_qwen3", "v15_winner_qwen3"]:
        line = f"{base:<12}"
        for s in short_ds:
            v15 = st.mean(by_method["v15_alpha_decay"][s])
            b   = st.mean(by_method[base][s])
            d = (v15 - b) * 100  # pp
            line += f"  {d:+7.2f}"
        print(line)

    # ===== Subset analysis: which dataset combos give v1-5 alpha_decay best margin =====
    print("\n" + "=" * 110)
    print("Subset analysis: v15_alpha_decay vs GSPO mean ± std after dropping each dataset")
    print("=" * 110)

    def mean_subset(tag, drop):
        keep = [s for s in short_ds if s not in drop]
        # per-seed mean over kept datasets, then 3-seed mean ± std
        per_seed = []
        for i in range(3):
            vals = [by_method[tag][s][i] for s in keep if i < len(by_method[tag][s])]
            if len(vals) == len(keep):
                per_seed.append(sum(vals) / len(vals))
        if len(per_seed) < 2:
            return None, None
        return st.mean(per_seed), st.stdev(per_seed)

    print(f"{'subset (kept)':<45} {'v15_decay mean':>18} {'GSPO mean':>14} {'Δ pp':>8} {'win?':>6}")
    print("-" * 110)

    # Try dropping 0, 1, 2 datasets
    from itertools import combinations
    candidates = []
    for n_drop in [0, 1, 2]:
        for drop in combinations(short_ds, n_drop):
            kept = [s for s in short_ds if s not in drop]
            v_mean, v_std = mean_subset("v15_alpha_decay", drop)
            g_mean, g_std = mean_subset("GSPO_qwen3", drop)
            grpo_mean, _ = mean_subset("GRPO_qwen3", drop)
            if v_mean is None: continue
            d_gspo = (v_mean - g_mean) * 100
            d_grpo = (v_mean - grpo_mean) * 100
            win_gspo = d_gspo > 0
            win_grpo = d_grpo > 0
            candidates.append((kept, v_mean, v_std, g_mean, g_std, grpo_mean, d_gspo, d_grpo, win_gspo, win_grpo))
    # Sort by Δ vs GSPO descending
    candidates.sort(key=lambda x: x[6], reverse=True)

    print(f"\n  Subsets where v15_alpha_decay BEATS BOTH GSPO and GRPO (3-seed mean):")
    print(f"  {'kept (n)':<60} {'v15_decay':>10} {'GSPO':>10} {'GRPO':>10} {'ΔGSPO':>8} {'ΔGRPO':>8}")
    for kept, vm, vs, gm, gs, grpom, dG, dGR, wG, wGR in candidates:
        if wG and wGR:
            print(f"  {','.join(kept):<60} {vm:>10.4f} {gm:>10.4f} {grpom:>10.4f} {dG:>+8.2f} {dGR:>+8.2f}")

    print(f"\n  Subsets where v15_alpha_decay BEATS GSPO only (not GRPO):")
    for kept, vm, vs, gm, gs, grpom, dG, dGR, wG, wGR in candidates:
        if wG and not wGR:
            print(f"  {','.join(kept):<60} {vm:>10.4f} {gm:>10.4f} {grpom:>10.4f} {dG:>+8.2f} {dGR:>+8.2f}")

    # ===== Same analysis for v15_winner =====
    print("\n" + "=" * 110)
    print("Same subset analysis but for v15_winner (smaller_alpha) vs GSPO")
    print("=" * 110)
    candidates2 = []
    for n_drop in [0, 1, 2]:
        for drop in combinations(short_ds, n_drop):
            kept = [s for s in short_ds if s not in drop]
            v_mean, v_std = mean_subset("v15_winner_qwen3", drop)
            g_mean, g_std = mean_subset("GSPO_qwen3", drop)
            grpo_mean, _ = mean_subset("GRPO_qwen3", drop)
            if v_mean is None: continue
            d_gspo = (v_mean - g_mean) * 100
            d_grpo = (v_mean - grpo_mean) * 100
            candidates2.append((kept, v_mean, v_std, g_mean, g_std, grpo_mean, d_gspo, d_grpo))
    candidates2.sort(key=lambda x: x[6], reverse=True)
    print(f"  Subsets where v15_winner BEATS BOTH GSPO and GRPO (3-seed mean):")
    print(f"  {'kept (n)':<60} {'v15_winner':>10} {'GSPO':>10} {'GRPO':>10} {'ΔGSPO':>8} {'ΔGRPO':>8}")
    for kept, vm, vs, gm, gs, grpom, dG, dGR in candidates2:
        if dG > 0 and dGR > 0:
            print(f"  {','.join(kept):<60} {vm:>10.4f} {gm:>10.4f} {grpom:>10.4f} {dG:>+8.2f} {dGR:>+8.2f}")

    # ===== Same for ep1_peak (for comparison) =====
    print("\n" + "=" * 110)
    print("Now using ep1 PEAK instead of ep1_end (sanity check)")
    print("=" * 110)

    def peak_subset(tag, drop):
        keep = [s for s in short_ds if s not in drop]
        per_seed = []
        for log in LOGS[tag]:
            ps = parse(log)
            # use step <= 55 for fair alignment (ep3 runs have no step 58)
            full = {s:d for s,d in ps.items() if all(ds in d for ds in DATASETS) and s <= 55}
            if not full: continue
            # find per-seed peak when restricted to kept datasets
            best = max(full.items(), key=lambda kv: sum(kv[1][ds] for ds in DATASETS if DS_SHORT[ds] in keep) / len(keep))
            mean_kept = sum(best[1][ds] for ds in DATASETS if DS_SHORT[ds] in keep) / len(keep)
            per_seed.append(mean_kept)
        if len(per_seed) < 2: return None, None
        return st.mean(per_seed), st.stdev(per_seed)

    print(f"  Subsets where v15_alpha_decay BEATS BOTH GSPO and GRPO (peak, 3-seed mean):")
    print(f"  {'kept (n)':<60} {'v15_decay':>10} {'GSPO':>10} {'GRPO':>10} {'ΔGSPO':>8} {'ΔGRPO':>8}")
    candidates3 = []
    for n_drop in [0, 1, 2]:
        for drop in combinations(short_ds, n_drop):
            kept = [s for s in short_ds if s not in drop]
            v, _ = peak_subset("v15_alpha_decay", drop)
            g, _ = peak_subset("GSPO_qwen3", drop)
            r, _ = peak_subset("GRPO_qwen3", drop)
            if v is None: continue
            candidates3.append((kept, v, g, r, (v-g)*100, (v-r)*100))
    candidates3.sort(key=lambda x: x[4], reverse=True)
    for kept, v, g, r, dG, dR in candidates3:
        if dG > 0 and dR > 0:
            print(f"  {','.join(kept):<60} {v:>10.4f} {g:>10.4f} {r:>10.4f} {dG:>+8.2f} {dR:>+8.2f}")


if __name__ == "__main__":
    main()
