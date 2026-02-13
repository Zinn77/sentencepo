#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict


def iter_jsonl_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
    else:
        files = [path]
    for f in files:
        yield f


def load_records(path):
    records = []
    for f in iter_jsonl_files(path):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _get_nested(record, key):
    if "." not in key:
        return record.get(key)
    cur = record
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def print_summary(records):
    corr = [r for r in records if r.get("reward") is not None and r.get("reward") > 0]
    wrong = [r for r in records if r.get("reward") is not None and r.get("reward") <= 0]

    def _avg(lst, key):
        vals = [v for v in (_get_nested(r, key) for r in lst) if v is not None]
        return sum(vals) / max(len(vals), 1)

    metrics = [
        "response_len",
        "sentence_count",
        "response_ppl",
        "response_entropy",
        "response_m",
        "response_delta",
        "sentence_stats.ppl_mean",
        "sentence_stats.ppl_var",
        "sentence_stats.entropy_mean",
        "sentence_stats.entropy_var",
    ]

    print("Total records:", len(records))
    if corr:
        print("Correct count:", len(corr))
        for m in metrics:
            print(f"Correct avg {m}:", _avg(corr, m))
    if wrong:
        print("Wrong count:", len(wrong))
        for m in metrics:
            print(f"Wrong avg {m}:", _avg(wrong, m))


def print_clip_stats(records):
    def _ratio(clipped_list):
        valid = [c for c in clipped_list if c is not None]
        if not valid:
            return None, 0
        return sum(1 for c in valid if c) / len(valid), len(valid)

    corr = [r for r in records if r.get("reward") is not None and r.get("reward") > 0]
    wrong = [r for r in records if r.get("reward") is not None and r.get("reward") <= 0]

    resp_clip_all = [
        r.get("response_clipped") if r.get("response_clipped") is not None else r.get("response_sentence_clipped")
        for r in records
    ]
    resp_clip_corr = [
        r.get("response_clipped") if r.get("response_clipped") is not None else r.get("response_sentence_clipped")
        for r in corr
    ]
    resp_clip_wrong = [
        r.get("response_clipped") if r.get("response_clipped") is not None else r.get("response_sentence_clipped")
        for r in wrong
    ]

    ratio_all, n_all = _ratio(resp_clip_all)
    ratio_corr, n_corr = _ratio(resp_clip_corr)
    ratio_wrong, n_wrong = _ratio(resp_clip_wrong)

    if ratio_all is not None:
        print("Response clip ratio:", f"{ratio_all:.4f}", f"(n={n_all})")
    if ratio_corr is not None:
        print("Correct response clip ratio:", f"{ratio_corr:.4f}", f"(n={n_corr})")
    if ratio_wrong is not None:
        print("Wrong response clip ratio:", f"{ratio_wrong:.4f}", f"(n={n_wrong})")

    sent_clip_all = []
    sent_clip_corr = []
    sent_clip_wrong = []
    for r in records:
        reward = r.get("reward")
        for s in r.get("sentences", []):
            if "clipped" not in s:
                continue
            sent_clip_all.append(s.get("clipped"))
            if reward is not None and reward > 0:
                sent_clip_corr.append(s.get("clipped"))
            elif reward is not None and reward <= 0:
                sent_clip_wrong.append(s.get("clipped"))

    ratio_all, n_all = _ratio(sent_clip_all)
    ratio_corr, n_corr = _ratio(sent_clip_corr)
    ratio_wrong, n_wrong = _ratio(sent_clip_wrong)

    if ratio_all is not None:
        print("Sentence clip ratio:", f"{ratio_all:.4f}", f"(n={n_all})")
    if ratio_corr is not None:
        print("Correct sentence clip ratio:", f"{ratio_corr:.4f}", f"(n={n_corr})")
    if ratio_wrong is not None:
        print("Wrong sentence clip ratio:", f"{ratio_wrong:.4f}", f"(n={n_wrong})")


def sentence_scatter(
    records,
    out_path=None,
    bins=None,
    xlim=None,
    ylim=None,
    color_by_clip=False,
):
    points = []
    rewards = []
    clipped_flags = []
    for r in records:
        reward = r.get("reward")
        for s in r.get("sentences", []):
            m = s.get("max_abs_log_ratio")
            delta = s.get("mean_log_ratio")
            if m is None or delta is None:
                continue
            points.append((m, abs(delta)))
            rewards.append(1 if reward is not None and reward > 0 else 0)
            clipped_flags.append(s.get("clipped"))

    if not points:
        print("No sentence points found.")
        return

    if out_path:
        try:
            import matplotlib.pyplot as plt

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.figure(figsize=(6, 4))
            if color_by_clip:
                clipped_x = [x for (x, _), c in zip(points, clipped_flags, strict=False) if c is True]
                clipped_y = [y for (_, y), c in zip(points, clipped_flags, strict=False) if c is True]
                unclipped_x = [x for (x, _), c in zip(points, clipped_flags, strict=False) if c is False]
                unclipped_y = [y for (_, y), c in zip(points, clipped_flags, strict=False) if c is False]
                other_x = [x for (x, _), c in zip(points, clipped_flags, strict=False) if c is None]
                other_y = [y for (_, y), c in zip(points, clipped_flags, strict=False) if c is None]

                if unclipped_x:
                    plt.scatter(unclipped_x, unclipped_y, s=4, alpha=0.4, label="unclipped")
                if clipped_x:
                    plt.scatter(clipped_x, clipped_y, s=4, alpha=0.6, label="clipped")
                if other_x:
                    plt.scatter(other_x, other_y, s=4, alpha=0.2, label="unknown")
                plt.legend(loc="best")
            else:
                plt.scatter(xs, ys, s=4, alpha=0.4)
            plt.xlabel("m = max |log ratio|")
            plt.ylabel("|Delta| = |mean log ratio|")
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            plt.title("Sentence-level m vs |Delta|")
            plt.tight_layout()
            plt.savefig(out_path)
            print("Saved scatter to", out_path)
        except Exception as e:
            print("Plotting failed:", e)

    if bins:
        bucket = defaultdict(lambda: [0, 0])
        for (m, _), r in zip(points, rewards, strict=False):
            for b in bins:
                if m <= b:
                    bucket[b][0] += r
                    bucket[b][1] += 1
                    break
            else:
                bucket["+"][0] += r
                bucket["+"][1] += 1
        print("m-bucket reward=1 ratios:")
        for b in sorted(bins):
            pos, total = bucket.get(b, (0, 0))
            print(f"  <= {b}: {pos / max(total, 1):.4f} (n={total})")
        if "+" in bucket:
            pos, total = bucket["+"]
            print(f"  <= +: {pos / max(total, 1):.4f} (n={total})")


def response_scatter(
    records,
    out_path=None,
    bins=None,
    delta_bins=None,
    xlim=None,
    ylim=None,
    color_by_clip=False,
):
    points = []
    rewards = []
    clipped_flags = []
    for r in records:
        reward = r.get("reward")
        sentences = r.get("sentences", [])
        if not sentences:
            continue
        m_vals = [s.get("max_abs_log_ratio") for s in sentences if s.get("max_abs_log_ratio") is not None]
        d_vals = [s.get("mean_log_ratio") for s in sentences if s.get("mean_log_ratio") is not None]
        if not m_vals or not d_vals:
            continue
        m = max(m_vals)
        delta = sum(d_vals) / len(d_vals)
        points.append((m, abs(delta)))
        rewards.append(1 if reward is not None and reward > 0 else 0)

        clip_flag = r.get("response_clipped")
        if clip_flag is None:
            clip_flag = r.get("response_sentence_clipped")
        clipped_flags.append(clip_flag)

        r["response_m"] = m
        r["response_delta"] = abs(delta)

    if not points:
        print("No response points found.")
        return

    if out_path:
        try:
            import matplotlib.pyplot as plt

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.figure(figsize=(6, 4))
            if color_by_clip:
                clipped_x = [x for (x, _), c in zip(points, clipped_flags, strict=False) if c is True]
                clipped_y = [y for (_, y), c in zip(points, clipped_flags, strict=False) if c is True]
                unclipped_x = [x for (x, _), c in zip(points, clipped_flags, strict=False) if c is False]
                unclipped_y = [y for (_, y), c in zip(points, clipped_flags, strict=False) if c is False]
                other_x = [x for (x, _), c in zip(points, clipped_flags, strict=False) if c is None]
                other_y = [y for (_, y), c in zip(points, clipped_flags, strict=False) if c is None]

                if unclipped_x:
                    plt.scatter(unclipped_x, unclipped_y, s=6, alpha=0.5, label="unclipped")
                if clipped_x:
                    plt.scatter(clipped_x, clipped_y, s=6, alpha=0.7, label="clipped")
                if other_x:
                    plt.scatter(other_x, other_y, s=6, alpha=0.2, label="unknown")
                plt.legend(loc="best")
            else:
                plt.scatter(xs, ys, s=6, alpha=0.5)
            plt.xlabel("response m = max |log ratio| over sentences")
            plt.ylabel("response |Delta| = |mean log ratio| over sentences")
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            plt.title("Response-level m vs |Delta|")
            plt.tight_layout()
            plt.savefig(out_path)
            print("Saved response scatter to", out_path)
        except Exception as e:
            print("Response plotting failed:", e)

    if bins:
        bucket = defaultdict(lambda: [0, 0])
        for (m, _), r in zip(points, rewards, strict=False):
            for b in bins:
                if m <= b:
                    bucket[b][0] += r
                    bucket[b][1] += 1
                    break
            else:
                bucket["+"][0] += r
                bucket["+"][1] += 1
        print("response m-bucket reward=1 ratios:")
        for b in sorted(bins):
            pos, total = bucket.get(b, (0, 0))
            print(f"  <= {b}: {pos / max(total, 1):.4f} (n={total})")
        if "+" in bucket:
            pos, total = bucket["+"]
            print(f"  <= +: {pos / max(total, 1):.4f} (n={total})")

    if delta_bins:
        bucket = defaultdict(lambda: [0, 0])
        for (_, d), r in zip(points, rewards, strict=False):
            for b in delta_bins:
                if d <= b:
                    bucket[b][0] += r
                    bucket[b][1] += 1
                    break
            else:
                bucket["+"][0] += r
                bucket["+"][1] += 1
        print("response |Delta|-bucket reward=1 ratios:")
        for b in sorted(delta_bins):
            pos, total = bucket.get(b, (0, 0))
            print(f"  <= {b}: {pos / max(total, 1):.4f} (n={total})")
        if "+" in bucket:
            pos, total = bucket["+"]
            print(f"  <= +: {pos / max(total, 1):.4f} (n={total})")


def plot_correct_wrong_compare(records, out_path, metrics):
    corr = [r for r in records if r.get("reward") is not None and r.get("reward") > 0]
    wrong = [r for r in records if r.get("reward") is not None and r.get("reward") <= 0]

    def _avg(lst, key):
        vals = [v for v in (_get_nested(r, key) for r in lst) if v is not None]
        return sum(vals) / max(len(vals), 1)

    corr_vals = [_avg(corr, m) for m in metrics]
    wrong_vals = [_avg(wrong, m) for m in metrics]

    try:
        import matplotlib.pyplot as plt

        x = range(len(metrics))
        width = 0.4
        plt.figure(figsize=(max(6, len(metrics) * 0.8), 4))
        plt.bar([i - width / 2 for i in x], corr_vals, width=width, label="correct")
        plt.bar([i + width / 2 for i in x], wrong_vals, width=width, label="wrong")
        plt.xticks(list(x), metrics, rotation=30, ha="right")
        plt.ylabel("mean value")
        plt.title("Correct vs Wrong response comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        print("Saved comparison plot to", out_path)
    except Exception as e:
        print("Comparison plot failed:", e)


def main():
    parser = argparse.ArgumentParser(description="View sentence analysis JSONL outputs")
    parser.add_argument("--input", required=True, help="Path to sentence_analysis dir or JSONL file")
    parser.add_argument("--uid", default=None, help="Filter by prompt uid")
    parser.add_argument("--scatter", default=None, help="Output PNG path for m vs |Delta| scatter")
    parser.add_argument(
        "--scatter-correct",
        default=None,
        help="Output PNG path for sentence scatter (reward=1)",
    )
    parser.add_argument(
        "--scatter-wrong",
        default=None,
        help="Output PNG path for sentence scatter (reward=0)",
    )
    parser.add_argument("--response-scatter", default=None, help="Output PNG path for response-level scatter")
    parser.add_argument(
        "--response-scatter-correct",
        default=None,
        help="Output PNG path for response-level scatter (reward=1)",
    )
    parser.add_argument(
        "--response-scatter-wrong",
        default=None,
        help="Output PNG path for response-level scatter (reward=0)",
    )
    parser.add_argument("--bins", default="64,128,256,512", help="Comma-separated m-bins")
    parser.add_argument("--response-bins", default=None, help="Comma-separated response m-bins")
    parser.add_argument("--response-delta-bins", default=None, help="Comma-separated response |Delta| bins")
    parser.add_argument(
        "--response-color-by-clip",
        action="store_true",
        help="Color response scatter by response_clipped (GSPO/GRPO) or response_sentence_clipped (SentencePO)",
    )
    parser.add_argument(
        "--sentence-color-by-clip",
        action="store_true",
        help="Color sentence scatter by sentence clipped (SentencePO)",
    )
    parser.add_argument("--compare-out", default=None, help="Output PNG path for correct/wrong comparison")
    parser.add_argument("--xlim", default=None, help="x-axis limits for scatter: min,max")
    parser.add_argument("--ylim", default=None, help="y-axis limits for scatter: min,max")
    parser.add_argument("--response-xlim", default=None, help="x-axis limits for response scatter: min,max")
    parser.add_argument("--response-ylim", default=None, help="y-axis limits for response scatter: min,max")
    parser.add_argument(
        "--compare-metrics",
        default=(
            "response_len,sentence_count,response_ppl,response_entropy,"
            "sentence_stats.ppl_mean,sentence_stats.ppl_var,"
            "sentence_stats.entropy_mean,sentence_stats.entropy_var"
        ),
        help="Comma-separated metric keys for comparison plot",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    if args.uid:
        records = [r for r in records if r.get("uid") == args.uid]

    if not records:
        print("No records found.")
        return

    response_bins = None
    if args.response_bins is not None:
        response_bins = [float(x) for x in args.response_bins.split(",") if x.strip()]
    response_delta_bins = None
    if args.response_delta_bins is not None:
        response_delta_bins = [float(x) for x in args.response_delta_bins.split(",") if x.strip()]

    bins = [float(x) for x in args.bins.split(",") if x.strip()]
    def _parse_lim(s):
        if not s:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("limit must be in 'min,max' format")
        return float(parts[0]), float(parts[1])

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    response_xlim = _parse_lim(args.response_xlim)
    response_ylim = _parse_lim(args.response_ylim)
    response_scatter(
        records,
        out_path=args.response_scatter,
        bins=response_bins,
        delta_bins=response_delta_bins,
        xlim=response_xlim,
        ylim=response_ylim,
        color_by_clip=args.response_color_by_clip,
    )
    if args.response_scatter_correct:
        response_scatter(
            [r for r in records if r.get("reward") is not None and r.get("reward") > 0],
            out_path=args.response_scatter_correct,
            bins=response_bins,
            delta_bins=response_delta_bins,
            xlim=response_xlim,
            ylim=response_ylim,
            color_by_clip=args.response_color_by_clip,
        )
    if args.response_scatter_wrong:
        response_scatter(
            [r for r in records if r.get("reward") is not None and r.get("reward") <= 0],
            out_path=args.response_scatter_wrong,
            bins=response_bins,
            delta_bins=response_delta_bins,
            xlim=response_xlim,
            ylim=response_ylim,
            color_by_clip=args.response_color_by_clip,
        )
    print_summary(records)
    print_clip_stats(records)
    sentence_scatter(
        records,
        out_path=args.scatter,
        bins=bins,
        xlim=xlim,
        ylim=ylim,
        color_by_clip=args.sentence_color_by_clip,
    )
    if args.scatter_correct:
        sentence_scatter(
            [r for r in records if r.get("reward") is not None and r.get("reward") > 0],
            out_path=args.scatter_correct,
            bins=bins,
            xlim=xlim,
            ylim=ylim,
            color_by_clip=args.sentence_color_by_clip,
        )
    if args.scatter_wrong:
        sentence_scatter(
            [r for r in records if r.get("reward") is not None and r.get("reward") <= 0],
            out_path=args.scatter_wrong,
            bins=bins,
            xlim=xlim,
            ylim=ylim,
            color_by_clip=args.sentence_color_by_clip,
        )

    if args.compare_out:
        metrics = [m.strip() for m in args.compare_metrics.split(",") if m.strip()]
        plot_correct_wrong_compare(records, args.compare_out, metrics)


if __name__ == "__main__":
    main()
