import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import datasets
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(raw_answer):
    boxed = last_boxed_only_string(str(raw_answer))
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except Exception:
            pass
    return str(raw_answer).strip()


def first_nonempty(example, keys):
    for key in keys:
        val = example.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        return val
    return None


def normalize_answer(answer):
    if answer is None:
        return None
    if isinstance(answer, dict):
        nested = first_nonempty(answer, ["answer", "solution", "final_answer", "value", "text"])
        if nested is not None:
            answer = nested
    if isinstance(answer, list):
        for item in reversed(answer):
            if item not in (None, ""):
                answer = item
                break
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="math-ai/minervamath")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/minerva",
        help="The save directory for the preprocessed dataset.",
    )
    args = parser.parse_args()

    loader_kwargs = {"split": args.split}
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, **loader_kwargs)
    else:
        dataset = datasets.load_dataset(args.dataset_name, **loader_kwargs)

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    def process_fn(example, idx):
        question = first_nonempty(
            example,
            [
                "question",
                "problem",
                "prompt",
                "instruction",
                "input",
                "content",
                "query",
                "text",
            ],
        )
        if not question:
            raise ValueError(f"Cannot find question field in {example.keys()}")
        question = question + " " + instruction_following

        raw_answer = normalize_answer(
            first_nonempty(
                example,
                [
                    "answer",
                    "solution",
                    "final_answer",
                    "output",
                    "target",
                    "label",
                    "response",
                ],
            )
        )
        if raw_answer is None:
            raise ValueError(f"Cannot find answer field in {example.keys()}")

        solution = extract_solution(raw_answer)

        data = {
            "data_source": args.dataset_name,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": args.split,
                "index": idx,
                "original_answer": raw_answer,
            },
        }
        return data

    dataset = dataset.map(process_fn, with_indices=True)
    dataset = dataset.select_columns(["data_source", "prompt",  "ability", "reward_model", "extra_info"])

    local_save_dir = args.local_dir or args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    parquet_name = f"{args.split}.parquet"
    dataset.to_parquet(os.path.join(local_save_dir, parquet_name))
    print(f"Saved to {local_save_dir}/{parquet_name}")
