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


def sentence_scatter(records, out_path=None, bins=None):
    points = []
    rewards = []
    for r in records:
        reward = r.get("reward")
        for s in r.get("sentences", []):
            m = s.get("max_abs_log_ratio")
            delta = s.get("mean_log_ratio")
            if m is None or delta is None:
                continue
            points.append((m, abs(delta)))
            rewards.append(1 if reward is not None and reward > 0 else 0)

    if not points:
        print("No sentence points found.")
        return

    if out_path:
        try:
            import matplotlib.pyplot as plt

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.figure(figsize=(6, 4))
            plt.scatter(xs, ys, s=4, alpha=0.4)
            plt.xlabel("m = max |log ratio|")
            plt.ylabel("|Delta| = |mean log ratio|")
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
                bucket["+"] [0] += r
                bucket["+"] [1] += 1
        print("m-bucket reward=1 ratios:")
        for k, (pos, total) in bucket.items():
            print(f"  <= {k}: {pos / max(total, 1):.4f} (n={total})")


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
    parser.add_argument("--bins", default="64,128,256,512", help="Comma-separated m-bins")
    parser.add_argument("--compare-out", default=None, help="Output PNG path for correct/wrong comparison")
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

    print_summary(records)

    bins = [float(x) for x in args.bins.split(",") if x.strip()]
    sentence_scatter(records, out_path=args.scatter, bins=bins)

    if args.compare_out:
        metrics = [m.strip() for m in args.compare_metrics.split(",") if m.strip()]
        plot_correct_wrong_compare(records, args.compare_out, metrics)


if __name__ == "__main__":
    main()
