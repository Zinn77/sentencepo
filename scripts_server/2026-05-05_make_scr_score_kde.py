"""
Generate the per-class SCR-score KDE figure that replaces the (unconvincing)
PCA panel in the paper. For each (layer, pooling) cell we compute the
class centroids c+/c- the same way the gap heatmap does --- mean over
rollout-level mean embeddings, then class-mean over rollouts, then L2 norm
--- and score every sentence by

    s_{i,k} = cos(h_{i,k}, c+) - cos(h_{i,k}, c-).

We then plot the per-class KDE of s on two panels: layer -1/last vs
layer -18/last. The figure also reports the per-class mean of s and the
AUROC of s for binary correct/wrong sentence classification, so the
caption can quote concrete discrimination numbers.

CPU-only, ~5s total. Writes <out_dir>/scr_score_kde.{pdf,png}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CB_ORANGE = "#E69F00"  # correct
CB_GREY = "#999999"     # wrong


def setup_style():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def normalise(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def class_centroids(emb: np.ndarray, correct: np.ndarray):
    """Build c+/c- the same way the gap heatmap does (per-sentence aggregation).

    c_pos_pre = mean of sent_emb over sentences in correct rollouts
    c_neg_pre = mean of sent_emb over sentences in wrong rollouts
    c+/c- = normalise(c_pos_pre) / normalise(c_neg_pre)

    Returns c+, c-, and the correct-sentence fraction.
    """
    pos_pre = emb[correct].mean(axis=0)
    neg_pre = emb[~correct].mean(axis=0)
    c_plus = normalise(pos_pre[None, :])[0]
    c_minus = normalise(neg_pre[None, :])[0]
    return c_plus, c_minus, float(correct.mean())


def scr_score(emb: np.ndarray, c_plus: np.ndarray, c_minus: np.ndarray) -> np.ndarray:
    h = normalise(emb)
    return h @ c_plus - h @ c_minus


def gaussian_kde_1d(x: np.ndarray, grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    """Plain Silverman-bandwidth Gaussian KDE in numpy."""
    n = x.size
    if bw is None:
        sd = x.std(ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
        bw = 0.9 * sigma * n ** (-1.0 / 5.0) if sigma > 0 else 1e-3
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC where label==True is the positive class."""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    pos = label
    neg = ~label
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rsum_pos = ranks[pos].sum()
    return float((rsum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_cell(artifacts: Path, layer: str, pooling: str):
    fp = artifacts / "sentence_embeddings" / f"layer_{layer}_pooling_{pooling}.pt"
    d = torch.load(fp, map_location="cpu", weights_only=False)
    emb = d["sent_emb"].float().numpy()
    sample = d["sent_sample"].numpy()
    correct = d["sent_correct"].numpy().astype(bool)
    return emb, sample, correct


def plot_panel(ax, score: np.ndarray, correct: np.ndarray, layer_label: str):
    s_pos = score[correct]
    s_neg = score[~correct]
    lo, hi = np.percentile(score, [0.5, 99.5])
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)

    kde_pos = gaussian_kde_1d(s_pos, grid)
    kde_neg = gaussian_kde_1d(s_neg, grid)

    ax.fill_between(grid, kde_neg, color=CB_GREY, alpha=0.45, lw=0,
                    label=f"wrong  (n={s_neg.size})")
    ax.fill_between(grid, kde_pos, color=CB_ORANGE, alpha=0.55, lw=0,
                    label=f"correct (n={s_pos.size})")
    ax.plot(grid, kde_neg, color=CB_GREY, lw=1.0)
    ax.plot(grid, kde_pos, color=CB_ORANGE, lw=1.0)

    ax.axvline(s_neg.mean(), color=CB_GREY, ls="--", lw=0.9, alpha=0.9)
    ax.axvline(s_pos.mean(), color=CB_ORANGE, ls="--", lw=0.9, alpha=0.9)

    delta = s_pos.mean() - s_neg.mean()
    auc = auroc(score, correct)
    ax.set_title(f"layer {layer_label} / last "
                 f"(Δmean={delta:+.4f}, AUROC={auc:.3f})")
    ax.set_xlabel(r"$s_{i,k} = \cos(h_{i,k}, c^+) - \cos(h_{i,k}, c^-)$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left")
    return delta, auc


def make_figure(artifacts: Path, out_dir: Path,
                layers=("-1", "-18"), pooling: str = "last"):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    summary = []
    for ax, layer in zip(axes, layers):
        emb, _sample, correct = load_cell(artifacts, layer, pooling)
        c_plus, c_minus, frac_correct = class_centroids(emb, correct)
        score = scr_score(emb, c_plus, c_minus)
        delta, auc = plot_panel(ax, score, correct, layer)
        summary.append((layer, delta, auc, frac_correct, score.size))
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "scr_score_kde.pdf"
    png = out_dir / "scr_score_kde.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("Wrote", pdf)
    print("Wrote", png)
    print("\nSummary:")
    print(f"{'layer':<8}{'delta_mean':>12}{'AUROC':>8}{'frac_correct':>14}{'n_sentences':>14}")
    for layer, delta, auc, fc, n in summary:
        print(f"{layer:<8}{delta:+12.5f}{auc:>8.3f}{fc:>14.3f}{n:>14}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", type=Path,
                   default=Path("CCdocs/2026-05-03_phaseA_diagnostic_artifacts"))
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--layers", nargs=2, default=["-1", "-18"])
    p.add_argument("--pooling", default="last")
    args = p.parse_args()
    out_dir = args.out_dir or (args.artifacts_dir / "figs")
    make_figure(args.artifacts_dir, out_dir, tuple(args.layers), args.pooling)


if __name__ == "__main__":
    main()
