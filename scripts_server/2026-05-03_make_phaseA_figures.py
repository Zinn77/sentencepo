"""Generate paper figures from Phase A diagnostic artifacts.

Inputs (default `CCdocs/2026-05-03_phaseA_diagnostic_artifacts/`):
    - sweep_metrics.csv          : 9 layer x 6 pooling rows + filtered_reason
    - sentence_embeddings/       : per-(layer, pooling) .pt with sent_emb / sent_correct
    - rollouts.jsonl, config.json: untouched (loaded only for sanity prints)

Outputs (under `<artifacts>/figs/`), each emitted as `.pdf` (paper) + `.png` (preview):
    - heatmap_layer_pooling_gap          9x6 heatmap, color = pos_neg_gap
    - layer_curve_gap                    x=layer (single only), y=gap, 6 lines = pooling
    - layer_curve_snr                    same x, y=scr_snr / slpa_snr (dual panel)
    - embedding_pca                      2 panel PCA on sent_emb for L-1 / L-18 (last pooling)
    - cos_global_filter                  scatter cos_global vs |c+|, mark diff pooling
    - gap_vs_snr_scatter                 scatter gap vs scr_snr, label points

Dependencies: numpy, matplotlib, torch (PCA via numpy SVD; CSV via stdlib).
CPU only; ~30s on the full artifact set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


# Layer order for x-axis on line plots: closest-to-output (-1) -> closest-to-input (-36).
SINGLE_LAYER_ORDER = ["-1", "-4", "-9", "-18", "-27", "-32", "-36"]
MULTI_LAYER_ORDER = ["-1|-9|-18", "-9|-18|-27"]
LAYER_ORDER = SINGLE_LAYER_ORDER + MULTI_LAYER_ORDER

POOLING_ORDER = ["last", "mean", "first", "mean_no_punct", "entropy_weighted", "diff"]
POOLING_LABEL = {
    "last": "Last",
    "mean": "Mean",
    "first": "First",
    "mean_no_punct": "Mean (no punct)",
    "entropy_weighted": "Entropy-W",
    "diff": "Diff",
}

# Color-blind safe palette (Tol's "bright" + Wong's points), diff is gray.
POOLING_COLOR = {
    "last":             "#0173b2",  # blue
    "mean":             "#de8f05",  # orange
    "first":            "#029e73",  # green
    "mean_no_punct":    "#cc78bc",  # purple
    "entropy_weighted": "#ca9161",  # brown
    "diff":             "#9a9a9a",  # gray (filtered)
}
CORRECT_COLOR = "#d55e00"  # vermillion
WRONG_COLOR = "#7a8aa8"    # muted slate-blue (legible against white, distinct from correct)


def layer_label(layer: str) -> str:
    """`-18` -> `L-18`, `-1|-9|-18` -> `L(-1,-9,-18)`."""
    if "|" in layer:
        return f"L({','.join(layer.split('|'))})"
    return f"L{layer}"


def layer_to_filename(layer: str) -> str:
    """CSV `-1|-9|-18` -> filename token `-1x-9x-18`."""
    return layer.replace("|", "x")


def load_metrics(csv_path: Path) -> list[dict[str, Any]]:
    """Read sweep_metrics.csv into a list of dicts; numeric fields cast to float."""
    numeric_cols = {
        "n_sentences", "n_correct", "n_wrong",
        "cos_global", "pos_neg_gap", "c_pos_norm", "c_neg_norm",
        "scr_score_var", "scr_snr", "slpa_score_var", "slpa_snr",
    }
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if k in numeric_cols:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = float("nan")
                else:
                    row[k] = v if v is not None else ""
            rows.append(row)
    return rows


def filter_rows(rows: list[dict[str, Any]], **eq) -> list[dict[str, Any]]:
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def find_row(rows: list[dict[str, Any]], **eq) -> dict[str, Any] | None:
    matches = filter_rows(rows, **eq)
    return matches[0] if matches else None


def pivot_metric(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    """Build a (len(LAYER_ORDER), len(POOLING_ORDER)) array of metric values, NaN if missing."""
    out = np.full((len(LAYER_ORDER), len(POOLING_ORDER)), np.nan, dtype=float)
    for i, layer in enumerate(LAYER_ORDER):
        for j, pooling in enumerate(POOLING_ORDER):
            r = find_row(rows, layer=layer, pooling=pooling)
            if r is not None:
                out[i, j] = r[metric]
    return out


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
            "pdf.fonttype": 42,  # editable text in PDF
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


def pca_2d(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    """Top-2 principal components via randomized SVD (torch.svd_lowrank).

    Full LAPACK SVD on (N=5k+, D=2560) gets killed under the sandbox memory cap;
    randomized lowrank is O(N*D*k) and finishes in <1s. Sign is arbitrary.
    """
    t = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
    centered = t - t.mean(dim=0, keepdim=True)
    torch.manual_seed(seed)
    u, s, _ = torch.svd_lowrank(centered, q=2, niter=4)
    return (u * s).numpy()


# ---- Figures ----------------------------------------------------------------


def plot_heatmap_gap(rows: list[dict[str, Any]], out_dir: Path) -> None:
    data = pivot_metric(rows, "pos_neg_gap")
    diff_idx = POOLING_ORDER.index("diff")

    # Color-scale on kept poolings only; Diff is filtered noise (cos~0) and
    # would otherwise compress the meaningful range (0.03 - 0.13) to one shade.
    kept_data = np.delete(data, diff_idx, axis=1)
    vmin = float(np.nanmin(kept_data))
    vmax = float(np.nanmax(kept_data))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    masked = np.ma.array(data, mask=np.zeros_like(data, dtype=bool))
    masked.mask[:, diff_idx] = True  # exclude Diff from imshow color mapping
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Render Diff column manually as solid gray with hatched overlay.
    for i in range(data.shape[0]):
        ax.add_patch(
            plt.Rectangle(
                (diff_idx - 0.5, i - 0.5), 1, 1,
                facecolor="#d9d9d9", edgecolor="white", hatch="///", linewidth=0.4,
            )
        )

    ax.set_xticks(np.arange(len(POOLING_ORDER)))
    ax.set_xticklabels([POOLING_LABEL[p] for p in POOLING_ORDER], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(LAYER_ORDER)))
    ax.set_yticklabels([layer_label(l) for l in LAYER_ORDER])
    ax.set_xlabel("Pooling")
    ax.set_ylabel("Hidden layer")
    ax.set_title("Pos-neg embedding gap (Phase A)")
    ax.grid(False)

    # Annotate each cell. Kept cells: text color flips on luminance; Diff: dark text.
    threshold = vmin + 0.55 * (vmax - vmin)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if j == diff_idx:
                color = "#404040"
            else:
                color = "white" if val < threshold else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("pos-neg gap (kept poolings)")
    # Mark the Diff column directly via the colorbar legend (small inset note).
    cbar.ax.annotate(
        "Diff: hatched\n(cos$\\approx$0,\n gap is noise)",
        xy=(0.5, 0.0), xycoords="axes fraction",
        xytext=(0.5, -0.18), textcoords="axes fraction",
        ha="center", va="top", fontsize=7, color="#404040",
    )
    save_fig(fig, out_dir, "heatmap_layer_pooling_gap")


def _line_plot(
    rows: list[dict[str, Any]],
    metric: str,
    ax: plt.Axes,
    title: str,
) -> None:
    x = np.arange(len(SINGLE_LAYER_ORDER))
    diff_y: np.ndarray | None = None
    kept_max = 0.0
    for pooling in POOLING_ORDER:
        y = np.array([
            (find_row(rows, layer=layer, pooling=pooling) or {}).get(metric, float("nan"))
            for layer in SINGLE_LAYER_ORDER
        ], dtype=float)
        if pooling == "diff":
            diff_y = y
        else:
            ax.plot(
                x, y,
                color=POOLING_COLOR[pooling],
                marker="o", linewidth=1.5, markersize=4.5,
                label=POOLING_LABEL[pooling],
            )
            kept_max = max(kept_max, float(np.nanmax(y)))

    # Clip y-axis to kept range; annotate Diff off-scale value as text.
    ax.set_ylim(0, kept_max * 1.18)
    if diff_y is not None and not np.all(np.isnan(diff_y)):
        diff_lo = float(np.nanmin(diff_y))
        diff_hi = float(np.nanmax(diff_y))
        ax.text(
            0.99, 0.97,
            f"Diff (filtered): {diff_lo:.2f}-{diff_hi:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=POOLING_COLOR["diff"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([layer_label(l) for l in SINGLE_LAYER_ORDER], rotation=0)
    ax.set_xlabel("Hidden layer (output -> input)")
    ax.set_title(title)


def plot_layer_curve_gap(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    _line_plot(rows, "pos_neg_gap", ax, "Pos-neg gap by layer")
    ax.set_ylabel("pos-neg gap")
    ax.legend(loc="best", ncol=2)
    save_fig(fig, out_dir, "layer_curve_gap")


def plot_layer_curve_snr(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharex=True)
    _line_plot(rows, "scr_snr", axes[0], "SCR signal-to-noise")
    axes[0].set_ylabel("SCR SNR")
    _line_plot(rows, "slpa_snr", axes[1], "SLPA signal-to-noise")
    axes[1].set_ylabel("SLPA SNR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, out_dir, "layer_curve_snr")


def plot_embedding_pca(
    artifacts_dir: Path,
    out_dir: Path,
    layers: tuple[str, str] = ("-1", "-18"),
    pooling: str = "last",
    rng_seed: int = 0,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=False)
    rng = np.random.default_rng(rng_seed)
    for ax, layer in zip(axes, layers):
        path = artifacts_dir / "sentence_embeddings" / f"layer_{layer_to_filename(layer)}_pooling_{pooling}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        emb = data["sent_emb"].to(torch.float32).numpy()
        correct = data["sent_correct"].numpy().astype(bool)

        # Subsample wrong points for legibility; keep all correct.
        wrong_idx = np.where(~correct)[0]
        if wrong_idx.size > 3000:
            wrong_idx = rng.choice(wrong_idx, size=3000, replace=False)
        correct_idx = np.where(correct)[0]
        keep = np.concatenate([wrong_idx, correct_idx])
        emb_sub = emb[keep]
        correct_sub = correct[keep]

        xy = pca_2d(emb_sub, seed=rng_seed)
        ax.scatter(
            xy[~correct_sub, 0], xy[~correct_sub, 1],
            s=7, c=WRONG_COLOR, alpha=0.35, edgecolors="none",
            label=f"wrong (n={int((~correct_sub).sum())})",
        )
        ax.scatter(
            xy[correct_sub, 0], xy[correct_sub, 1],
            s=9, c=CORRECT_COLOR, alpha=0.75, edgecolors="none",
            label=f"correct (n={int(correct_sub.sum())})",
        )
        ax.set_title(f"{layer_label(layer)} / {POOLING_LABEL[pooling]}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best")
    fig.suptitle("Sentence-embedding PCA: shallow vs deep layer", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "embedding_pca")


def plot_cos_global_filter(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for pooling in POOLING_ORDER:
        sub = filter_rows(rows, pooling=pooling)
        xs = np.array([r["cos_global"] for r in sub], dtype=float)
        ys = np.array([r["c_pos_norm"] for r in sub], dtype=float)
        is_diff = pooling == "diff"
        ax.scatter(
            xs, ys,
            s=46, color=POOLING_COLOR[pooling],
            marker="x" if is_diff else "o",
            alpha=0.6 if is_diff else 0.9,
            edgecolors="none" if not is_diff else None,
            label="Diff (filtered)" if is_diff else POOLING_LABEL[pooling],
            linewidths=1.5 if is_diff else 0,
        )
    # Highlight excluded bands.
    ax.axvspan(-0.05, 0.05, color="#f4cccc", alpha=0.4, lw=0)
    ax.axvspan(0.95, 1.05, color="#f4cccc", alpha=0.4, lw=0)
    ax.text(0.0, ax.get_ylim()[1] * 0.95, "filtered\n(cos<0.05)", fontsize=8, ha="center", color="#a04040")
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel("cos_global (mean off-diagonal cosine)")
    ax.set_ylabel(r"$\|c_+\|$ (positive-center norm)")
    ax.set_title(r"Filter: cos_global $\notin$ [0.05, 0.95]")
    ax.legend(loc="best", ncol=2, fontsize=8)
    save_fig(fig, out_dir, "cos_global_filter")


def plot_gap_vs_snr_scatter(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    kept = [r for r in rows if r.get("filtered_reason", "") == ""]
    filt = [r for r in rows if r.get("filtered_reason", "") != ""]
    for pooling in POOLING_ORDER:
        if pooling == "diff":
            continue
        sub = filter_rows(kept, pooling=pooling)
        xs = np.array([r["pos_neg_gap"] for r in sub], dtype=float)
        ys = np.array([r["scr_snr"] for r in sub], dtype=float)
        ax.scatter(
            xs, ys,
            s=46, color=POOLING_COLOR[pooling],
            label=POOLING_LABEL[pooling],
            edgecolors="white", linewidths=0.5,
        )
    fxs = np.array([r["pos_neg_gap"] for r in filt], dtype=float)
    fys = np.array([r["scr_snr"] for r in filt], dtype=float)
    ax.scatter(
        fxs, fys,
        s=36, color=POOLING_COLOR["diff"],
        marker="x", alpha=0.55,
        label="Diff (filtered)", linewidths=1.5,
    )

    # Highlight the dual top-1 (-18 / last) and the SNR champion (-1 / last).
    for layer, marker_label in [("-18", "L-18 / Last"), ("-1", "L-1 / Last")]:
        row = find_row(rows, layer=layer, pooling="last")
        if row is None:
            continue
        ax.annotate(
            marker_label,
            xy=(row["pos_neg_gap"], row["scr_snr"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.6},
        )

    ax.set_xlabel("pos-neg gap")
    ax.set_ylabel("SCR SNR")
    ax.set_title("Gap vs SNR (kept cells)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    save_fig(fig, out_dir, "gap_vs_snr_scatter")


# ---- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=Path("CCdocs/2026-05-03_phaseA_diagnostic_artifacts"),
        help="Directory holding sweep_metrics.csv and sentence_embeddings/.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Where to drop figures (default: <artifacts>/figs/).",
    )
    parser.add_argument(
        "--pca_layers",
        nargs=2,
        default=("-1", "-18"),
        help="Two layer ids for the PCA panel (default: -1 -18).",
    )
    parser.add_argument(
        "--pca_pooling",
        default="last",
        help="Pooling for the PCA panel (default: last).",
    )
    args = parser.parse_args()

    artifacts_dir: Path = args.artifacts_dir.resolve()
    out_dir: Path = (args.out_dir or (artifacts_dir / "figs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "sweep_metrics.csv"
    rows = load_metrics(csv_path)

    cfg_path = artifacts_dir / "config.json"
    cfg_brief: dict[str, Any] = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for k in ("model", "model_path", "tp_size", "n", "max_response", "git_commit"):
            if k in cfg:
                cfg_brief[k] = cfg[k]
    n_total = len(rows)
    n_kept = sum(1 for r in rows if r.get("filtered_reason", "") == "")
    print(f"Loaded {n_total} cells from {csv_path} (kept={n_kept}, filtered={n_total - n_kept})")
    if cfg_brief:
        print(f"Config: {cfg_brief}")

    setup_style()

    plot_heatmap_gap(rows, out_dir)
    plot_layer_curve_gap(rows, out_dir)
    plot_layer_curve_snr(rows, out_dir)
    plot_embedding_pca(artifacts_dir, out_dir, layers=tuple(args.pca_layers), pooling=args.pca_pooling)
    plot_cos_global_filter(rows, out_dir)
    plot_gap_vs_snr_scatter(rows, out_dir)

    print(f"Figures written to: {out_dir}")
    for stem in [
        "heatmap_layer_pooling_gap",
        "layer_curve_gap",
        "layer_curve_snr",
        "embedding_pca",
        "cos_global_filter",
        "gap_vs_snr_scatter",
    ]:
        for ext in ("pdf", "png"):
            p = out_dir / f"{stem}.{ext}"
            print(f"  {'OK' if p.exists() else 'MISS'}  {p.name}")


if __name__ == "__main__":
    main()
