#!/usr/bin/env python
"""Generate 8 Phase B run scripts (2 modules x 4 experiments) for Hope platform.

Reads Phase A winners (hidden_layer + pooling for SCR and SLPA top-2 each)
and emits self-contained shell scripts that can be uncommented in
jupyter.sh and submitted via `hope run` in parallel.

Usage:
    python scripts_server/gen_phaseB_scripts.py \
        --output_dir /root/scripts_server/运行脚本_0427 \
        --remote_repo /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02 \
        --scr_top1_layer -9 --scr_top1_pool mean \
        --scr_top2_layer -1 --scr_top2_pool mean_no_punct \
        --slpa_top1_layer -9 --slpa_top1_pool mean \
        --slpa_top2_layer "-1|-9|-18" --slpa_top2_pool mean
"""
from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = """#!/usr/bin/env bash
set -ex

# v1-5-hidden Phase B: {tag}
# MODULE={module}  LAYER={layer}  POOLING={pooling}  ALPHA={alpha}
# (Generated; see test_v1-5_hidden_phaseB.sh for the full param list.)

export MODULE={module}
export LAYER='{layer}'
export POOLING={pooling}
export ALPHA={alpha}
export SEED={seed}
export EPOCHS={epochs}
export EXP_TAG_OVERRIDE={tag}

cd {remote_repo}/sentencepo_v1-5
bash test_v1-5_hidden_phaseB.sh
"""


def emit(out_dir: Path, name: str, **kwargs) -> Path:
    path = out_dir / f"{name}.sh"
    path.write_text(TEMPLATE.format(**kwargs))
    path.chmod(0o755)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--remote_repo", required=True, help="platform path prefix where sentencepo_v1-5 lives")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    # SCR
    p.add_argument("--scr_top1_layer", required=True)
    p.add_argument("--scr_top1_pool", required=True)
    p.add_argument("--scr_top2_layer", required=True)
    p.add_argument("--scr_top2_pool", required=True)
    # SLPA
    p.add_argument("--slpa_top1_layer", required=True)
    p.add_argument("--slpa_top1_pool", required=True)
    p.add_argument("--slpa_top2_layer", required=True)
    p.add_argument("--slpa_top2_pool", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    common = dict(
        alpha=args.alpha,
        seed=args.seed,
        epochs=args.epochs,
        remote_repo=args.remote_repo,
    )

    runs: list[Path] = []
    # Baseline (no module).
    runs.append(emit(out, "phaseB_01_baseline", module="none", layer="-1", pooling="last", tag="phaseB_baseline", **common))
    # Current SCR.
    runs.append(emit(out, "phaseB_02_scr_current", module="scr", layer="-1", pooling="last", tag="phaseB_scr_current", **common))
    runs.append(emit(out, "phaseB_03_scr_top1", module="scr", layer=args.scr_top1_layer, pooling=args.scr_top1_pool, tag="phaseB_scr_top1", **common))
    runs.append(emit(out, "phaseB_04_scr_top2", module="scr", layer=args.scr_top2_layer, pooling=args.scr_top2_pool, tag="phaseB_scr_top2", **common))
    # Current SLPA.
    runs.append(emit(out, "phaseB_05_slpa_current", module="slpa", layer="-1", pooling="last", tag="phaseB_slpa_current", **common))
    runs.append(emit(out, "phaseB_06_slpa_top1", module="slpa", layer=args.slpa_top1_layer, pooling=args.slpa_top1_pool, tag="phaseB_slpa_top1", **common))
    runs.append(emit(out, "phaseB_07_slpa_top2", module="slpa", layer=args.slpa_top2_layer, pooling=args.slpa_top2_pool, tag="phaseB_slpa_top2", **common))
    # Spare slot intentionally identical to baseline (drop or replace as needed).
    runs.append(emit(out, "phaseB_08_baseline_extra", module="none", layer="-1", pooling="last", tag="phaseB_baseline_seed", **common))

    print("Generated:")
    for r in runs:
        print(f"  {r}")
    print()
    print("To submit, for each script:")
    print("  1. Edit /root/scripts_server/hope_dir_0213/jupyter.sh and uncomment ONE script line")
    print("  2. hope run /root/scripts_server/hope_dir_0213/run.hope")
    print("  3. Repeat for parallel runs")


if __name__ == "__main__":
    main()
