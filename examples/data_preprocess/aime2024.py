import argparse
import os
import re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_aime_answer(raw_answer):
    """
    AIME 答案都是 0-999 的整数。
    处理各种格式：
    - \boxed{123}
    - \textbf{(123)}
    - 完整解答文本（提取最后的数字）
    - 纯数字
    """
    raw_answer = str(raw_answer).strip()
    
    # 1. 尝试提取 \boxed{}
    boxed = last_boxed_only_string(raw_answer)
    if boxed is not None:
        try:
            content = remove_boxed(boxed)
            # 从 boxed 内容中提取数字
            nums = re.findall(r'\d+', content)
            if nums:
                return nums[-1]
            return content.strip()
        except Exception:
            pass
    
    # 2. 提取 \textbf{(xxx)} 或 \textbf{xxx} 中的数字
    textbf_match = re.search(r'\\textbf\{\s*\(?\s*(\d+)\s*\)?\s*\}', raw_answer)
    if textbf_match:
        return textbf_match.group(1)
    
    # 3. 如果是短字符串且只包含数字，直接返回
    cleaned = re.sub(r'[^\d]', '', raw_answer)
    if len(raw_answer) < 20 and cleaned.isdigit():
        return cleaned
    
    # 4. 如果答案很长（完整解答），从文本中提取 AIME 格式的答案
    #    AIME 答案通常在末尾，格式如 "the answer is 123" 或 "= 123"
    if len(raw_answer) > 50:
        # 尝试匹配常见的答案模式
        patterns = [
            r'(?:answer|Answer|ANSWER)\s*(?:is|=|:)\s*\$?\\?boxed\{?(\d+)\}?\$?',
            r'(?:answer|Answer|ANSWER)\s*(?:is|=|:)\s*\$?(\d+)\$?',
            r'=\s*\$?\\?boxed\{(\d+)\}\$?\s*$',
            r'=\s*(\d+)\s*$',
            r'\$(\d{1,3})\$\s*\.?\s*$',  # 末尾的 $123$
        ]
        for pattern in patterns:
            match = re.search(pattern, raw_answer)
            if match:
                return match.group(1)
        
        # 如果没有匹配到特定模式，取最后出现的1-3位数字
        all_nums = re.findall(r'\b(\d{1,3})\b', raw_answer)
        if all_nums:
            return all_nums[-1]
    
    # 5. 最后尝试：提取任何数字
    nums = re.findall(r'\d+', raw_answer)
    if nums:
        return nums[-1]
    
    # 6. 无法提取，返回清理后的原始字符串
    return raw_answer.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_save_dir", default="~/data/aime2024", help="The save directory for the preprocessed dataset.")
    args = parser.parse_args()

    data_source = "HuggingFaceH4/aime_2024"
    print(f"Loading {data_source}...", flush=True)
    
    # AIME 2024 通常只有 train split，但我们把它作为测试集使用
    dataset = datasets.load_dataset(data_source, split="train")

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    def process_fn(example, idx):
        # 尝试获取 problem/question 字段
        question = example.get("problem") or example.get("question")
        if not question:
            raise ValueError(f"Cannot find question field in {example.keys()}")
            
        question = question + " " + instruction_following

        # 尝试获取 solution/answer 字段
        answer = example.get("solution") or example.get("answer")
        if not answer:
            raise ValueError(f"Cannot find answer field in {example.keys()}")

        solution = extract_aime_answer(answer)
        
        data = {
            "data_source": "aime2024",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "test", "index": idx, "original_answer": answer},
        }
        return data

    # 处理数据
    dataset = dataset.map(process_fn, with_indices=True)
    # 只保留需要的列，移除原始数据集的其他字段
    dataset = dataset.select_columns(["data_source", "prompt",  "ability", "reward_model", "extra_info"])

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # 保存为 test.parquet (因为 AIME 是用来评测的)
    dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    
    print(f"Saved to {local_save_dir}/test.parquet")
