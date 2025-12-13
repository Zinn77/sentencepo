import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    boxed = last_boxed_only_string(solution_str)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except Exception:
            pass
    return str(solution_str).strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_save_dir", default="~/data/amc23", help="The save directory for the preprocessed dataset.")
    args = parser.parse_args()

    data_source = "math-ai/amc23"
    print(f"Loading {data_source}...", flush=True)
    
    # AMC 23 可能在 train 或 test split
    try:
        dataset = datasets.load_dataset(data_source, split="test")
    except:
        dataset = datasets.load_dataset(data_source, split="train")

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example.get("problem") or example.get("question")
        if not question:
            raise ValueError(f"Cannot find question field in {example.keys()}")
            
        question = question + " " + instruction_following

        answer = example.get("solution") or example.get("answer")
        if not answer:
            raise ValueError(f"Cannot find answer field in {example.keys()}")

        solution = extract_solution(answer)
        
        data = {
            "data_source": "amc23",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "test", "index": idx, "original_answer": answer},
        }
        return data

    dataset = dataset.map(process_fn, with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"Saved to {local_save_dir}/test.parquet")
