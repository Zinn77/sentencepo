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

def load_aime2025_all(data_source="opencompass/AIME2025", split="train"):
    ds_i = datasets.load_dataset(data_source, name="AIME2025-I", split=split)
    ds_ii = datasets.load_dataset(data_source, name="AIME2025-II", split=split)
    return datasets.concatenate_datasets([ds_i, ds_ii])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_save_dir", default="~/data/aime2025", help="The save directory for the preprocessed dataset.")
    args = parser.parse_args()

    data_source = "opencompass/AIME2025"
    print(f"Loading {data_source} (I+II)...", flush=True)
    
    # 尝试加载 train split，如果不存在则加载 test
    try:
        dataset = load_aime2025_all(split="train")
    except Exception:
        dataset = load_aime2025_all(split="test")

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    def process_fn(example, idx):
        # 适配不同的字段名
        question = example.get("problem") or example.get("question") or example.get("input")
        if not question:
            raise ValueError(f"Cannot find question field in {example.keys()}")
            
        question = question + " " + instruction_following

        answer = example.get("solution") or example.get("answer") or example.get("target")
        if not answer:
            raise ValueError(f"Cannot find answer field in {example.keys()}")

        solution = extract_solution(answer)
        
        data = {
            "data_source": "aime2025",
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
