import argparse
import json
import torch
import tqdm
import os

import re
import signal
from typing import Optional
import numpy as np
from typing import Union
from latex2sympy2 import latex2sympy

from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question, without destroying separators."""
    # 仅保留等号右侧
    final_answer = final_answer.split("=")[-1]

    # 轻量替换：去掉明显无关的 LaTeX 包裹，但不要去空格/逗号
    for before, after in [
        ("mbox", "text"),
        (",$\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
    ]:
        final_answer = final_answer.replace(before, after)

    for expr in [
        "\\ldots", "\\dots", "\\text{}^2", "\\text{}^3",
        "\\text{.}", "\\text{ }", "\\text{}"
    ]:
        final_answer = final_answer.replace(expr, "")

    # 去掉 \left \right 与常见括号包裹
    final_answer = re.sub(r"\\left\s*\(", "(", final_answer)
    final_answer = re.sub(r"\\right\s*\)", ")", final_answer)
    final_answer = re.sub(r"\\left\s*\[", "[", final_answer)
    final_answer = re.sub(r"\\right\s*\]", "]", final_answer)
    final_answer = final_answer.replace(r"\left", "").replace(r"\right", "")

    # 仅保留第一段数学内容（若有 $...$），否则保留原文
    m = re.search(r"\$(.*?)\$", final_answer)
    if m:
        final_answer = m.group(1)

    # 去掉 \text{...} / \textbf{...} / \overline{...} 的壳
    final_answer = re.sub(r"\\textbf\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\text\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\overline\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\boxed\{(.*)\}", r"\1", final_answer)

    # 规范分数/开方等简写
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)

    # ——关键改动——
    # 仅移除“确认为千分位”的逗号：数字后跟逗号，逗号后紧跟3位数字并以非数字或结尾收束
    final_answer = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", final_answer)

    # 去掉首尾空白；不要移除中间空格/逗号
    return final_answer.strip()

################################

def verify_answer_predict(generated_answer, final_ground_truth):
    '''predict approxiamate value of accurate answer'''
    final_answer = extract_answer(generated_answer)
    if final_answer == "NA":
        return -1.0, False, final_answer  # r acc pred
    final_answer = normalize_final_answer(final_answer)
    temp = final_answer
    final_answer = extract_numeric_value(final_answer)
    if final_answer == "Format Error":
        return -1.0, False, temp

    # 计算误差百分比（绝对值）
    error_percentage = (abs(final_answer - final_ground_truth) / (abs(final_ground_truth)+1e-7)) * 100

    # 判断是否在20%范围内
    is_correct = error_percentage <= 20

    # 计算奖励值：在20%范围内时，奖励从0到1线性变化
    # 当完全准确时奖励为1，当正好在20%边界上时奖励为0
    if is_correct:
        reward = 1.0 - (error_percentage / 20.0)
    else:
        reward = -1.0

    return reward, is_correct, temp

def extract_answer(answer):
    if "\\boxed" not in answer:
        return "NA"

    after_boxed = answer.split("\\boxed")[-1]
    paren_stack = []
    start_found = False
    start_index = None
    end_index = None
    for i, char in enumerate(after_boxed):
        if char == "{":
            if not start_found:
                start_index = i
                start_found = True
            paren_stack.append(char)
        elif char == "}":
            if paren_stack:
                paren_stack.pop()
            if start_found and len(paren_stack) == 0:
                end_index = i
                break
    if start_index is not None and end_index is not None:
        return after_boxed[start_index+1:end_index]
    else:
        return "NA"

def extract_numeric_value(input_str):
    match = re.search(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', input_str)

    if not match:
        raise ValueError("输入字符串中未找到数值部分")

    numeric_str = match.group(1)
    numeric_value = float(numeric_str)

    if '%' in input_str:
        return numeric_value / 100.0
    else:
        return numeric_value

#
#gt = 130
#ans = '<think> 122% </think> \\boxed{130 Mpa}'
#print(compute_score(ans, gt, data_source = "material10k_magnitude"))
    

#################################

# ==============================================================================
# 主评测逻辑
# ==============================================================================
def main(args):
    """
    使用 vLLM 进行批量推理和 pass@k 评估。
    该脚本会迭代最多 k 次（使用不同的种子），并从后续的运行中移除已经成功解决的问题。
    """
    model_name = os.path.basename(os.path.normpath(args.model_path))
    print("="*20 + f" 开始为模型进行 Pass@{args.k} 评估: {model_name} " + "="*20)

    # 1. 解析种子列表
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        if len(seeds) < args.k:
            raise ValueError(f"提供的种子数量 ({len(seeds)}) 少于 k ({args.k})。请提供至少 k 个种子。")
    except Exception as e:
        print(f"解析种子时出错: {e}")
        return

    # 2. 加载模型和分词器
    print(f"从 {args.model_path} 加载模型和分词器...")
    try:
        # 注意：为了可复现性，这里的 seed 参数在 vLLM 中用于初始化分布式环境，
        # 我们将在每次循环中通过 sampling_params 传递不同的种子以改变生成结果。
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            seed=args.seed  # 基础初始化种子
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print("模型和分词器加载成功。")
    except Exception as e:
        print(f"加载模型或分词器失败: {e}")
        return

    # 3. 加载数据集
    print(f"从 {args.input_file} 加载 JSON 数据...")
    try:
        dataset = Dataset.from_json(args.input_file)
        total_count = len(dataset)
        print(f"成功加载 {total_count} 条记录。")
    except Exception as e:
        print(f"加载 JSON 文件失败: {e}")
        return


    # 4. 初始化结果追踪
    all_results = {}
    for i in range(len(dataset)):
        item_id = dataset[i]["id"]
        question_text = dataset[i]["question"]
        ground_truth = dataset[i]["ideal_answer"]

        all_results[item_id] = {
            "id": item_id,
            "prompt": [{"role": "user", "content": question_text}],
            "ground_truth": ground_truth,
            "is_correct_in_k": False,
            "solved_at_attempt": -1,
            "generations": []
        }

    unsolved_ids = set(all_results.keys())
    correct_in_k_count = 0
    batch_size = args.batch_size

    # 5. Pass@k 循环
    for attempt in range(1, args.k + 1):
        current_seed = int(args.seeds.split(',')[attempt - 1])
        print(f"\n--- 第 {attempt}/{args.k} 次尝试 | 使用种子: {current_seed} ---")

        if not unsolved_ids:
            print("🎉 所有问题均已解决！提前终止评估。")
            break

        unsolved_dataset = [d for d in dataset if d["id"] in unsolved_ids]
        print(f"待解决问题数量: {len(unsolved_dataset)}")

        # 配置采样参数
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            skip_special_tokens=True,
        )

        newly_solved_in_this_attempt = 0

        # 批处理循环
        for i in tqdm.tqdm(range(0, len(unsolved_dataset), batch_size), desc=f"第 {attempt} 次尝试处理中"):
            batch_data = unsolved_dataset[i:i + batch_size]

            batch_prompts, batch_ids, batch_ground_truths = [], [], []
            for item in batch_data:
                prompt_template = r"""You are a materials science expert. Please answer the following question, and put the final answer inside the LaTeX symbol \boxed{}. The output inside \boxed{} must adhere to the following unit conventions:
- **Yield strength, tensile strength, or similar:** Numerical value (without units) if the answer is in MPa.
- **Elongation or similar:** Percentage value (with the '%' symbol).
- **Hardness or similar:** Numerical value (without units) if the answer is in HV.

Do not include units such as 'MPa','HV', or any other text in your response. Only provide the true value as specified in the ground truth. For example:
- If the ground truth is '120 MPa', output: \boxed{120}
- If the ground truth is '17.0%', output: \boxed{17.0%}
- If the ground truth is '200 HV', output: \boxed{200}
- All answers that are percentages must include the percent sign, e.g., {18\\%}.
For the remaining data types, please use common units.

Here's the format request:
Please provide your reasoning process enclosed within <think> and </think> tags, followed by your final answer with explanation. i.e., <think> your reasoning here </think> your answer here, including \boxed{final answer}
Please ensure your output is as concise as possible while maintaining accuracy and reliability. When answering, beyond recalling and remembering inherent material properties, focus more on the application of these properties and logical scientific derivation.
Let's think step by step.
"""

                batch_prompts.append(prompt_template + "\n\n" + item["question"])
                batch_ids.append(item["id"])
                batch_ground_truths.append(item["ideal_answer"])

            # 跳过空批次
            if not batch_prompts:
                continue

            # 生成答案
            batch_outputs = llm.generate(batch_prompts, sampling_params)

            for j, request_output in enumerate(batch_outputs):
                item_id = batch_ids[j]
                ground_truth = batch_ground_truths[j]
                generated_answer = request_output.outputs[0].text.strip()

                try:
                    # 统一用 extract_numeric_value 来处理 ground_truth
                    try:
                        gt_value = extract_numeric_value(ground_truth)
                    except Exception as e:
                        print(f"Ground truth {ground_truth} 解析失败: {e}")
                        gt_value = float(re.search(r'([+-]?\d+\.?\d*)', ground_truth).group(1))

                    score, is_correct, model_prediction = verify_answer_predict(generated_answer, gt_value)

                except Exception as e:
                    print(f"\n警告: ID {item_id} 调用 verify_answer_predict 时出错: {e}")
                    is_correct = False
                    model_prediction = f"EVALUATION_ERROR: {e}"

                # 记录生成结果
                generation_data = {
                    "attempt": attempt,
                    "seed": current_seed,
                    "generated_answer": generated_answer,
                    "model_prediction": model_prediction,
                    "is_correct": bool(is_correct)
                }
                all_results[item_id]["generations"].append(generation_data)

                # 更新已解决状态
                if is_correct and not all_results[item_id]["is_correct_in_k"]:
                    all_results[item_id]["is_correct_in_k"] = True
                    all_results[item_id]["solved_at_attempt"] = attempt
                    unsolved_ids.remove(item_id)
                    correct_in_k_count += 1
                    newly_solved_in_this_attempt += 1

        print(f"第 {attempt} 次尝试完成。新解决问题数: {newly_solved_in_this_attempt}")
        current_accuracy = correct_in_k_count / len(dataset)
        print(f"当前 Pass@{args.k} 准确率: {current_accuracy:.4f} ({correct_in_k_count}/{len(dataset)})")

        # 6. 保存中间结果
        # 从最终输出文件名派生中间文件名
        output_dir = os.path.dirname(args.output_file)
        base_name = os.path.basename(args.output_file).replace('.json', '')
        intermediate_output_path = os.path.join(output_dir, f"{base_name}_attempt_{attempt}.json")

        print(f"保存第 {attempt} 次尝试的中间结果到 {intermediate_output_path}...")
        try:
            # 创建中间输出的副本，以便不修改主循环中的数据结构
            intermediate_output = {
                "model_name": model_name,
                "pass_at_k_configuration": args.k,
                "current_attempt": attempt,
                "current_seed": current_seed,
                "current_pass_rate": current_accuracy,
                "total_samples": total_count,
                "correct_samples_so_far": correct_in_k_count,
                "results": list(all_results.values()) # 转换为列表以便序列化
            }
            with open(intermediate_output_path, 'w', encoding='utf-8') as f:
                json.dump(intermediate_output, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存中间结果文件失败: {e}")


    # 7. 计算最终准确率并准备最终输出文件
    final_accuracy = (correct_in_k_count / total_count) if total_count > 0 else 0
    print("\n" + "="*20 + " 评估完成 " + "="*20)
    print(f"模型: {model_name}")
    print(f"总样本数: {total_count}")
    print(f"Pass@{args.k} 正确数: {correct_in_k_count}")
    print(f"最终 Pass@{args.k} 准确率: {final_accuracy:.4f}")

    final_output = {
        "model_name": model_name,
        "pass_at_k": args.k,
        "accuracy": final_accuracy,
        "total_samples": total_count,
        "correct_samples": correct_in_k_count,
        "results": list(all_results.values()) # 转换为列表以便序列化
    }

    # 8. 保存最终结果到 JSON 文件
    print(f"正在保存最终评估结果到 {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        print(f"所有任务已完成！评估结果已成功保存到 {args.output_file}")
    except Exception as e:
        print(f"保存最终结果文件失败: {e}")


if __name__ == "__main__":
    # 保证 vLLM 可复现性
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    parser = argparse.ArgumentParser(description="使用 vLLM 对语言模型进行高效的 pass@k 评估。")
    
    # 路径参数
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/gemini/code/grpo/Qwen2.5-7B-Instruct-GRPO_step86140", 
        # default="/mnt-nfsdata/MaterialCode/base-model/DeepSeek-R1-0528-Qwen3-8B",
        help="待评估的基础 LLM 模型路径"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="/mnt-nfsdata/zhengye/qna_dataset_pipeline/setting1/final_output/0723_val_dataset_400.json", 
        help="输入 .json 文件路径"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="/gemini/code/outputs/baseline_results.json", 
        help="最终输出的 .json 结果文件路径"
    )

    # vLLM 性能参数
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="vLLM 的张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="vLLM 的 GPU 内存利用率")
    parser.add_argument("--batch_size", type=int, default=64, help="推理的批处理大小")

    # 模型生成与评估参数
    parser.add_argument("--k", "--pass_at_k", type=int, default=1, help="pass@k 评估中每个问题的最大尝试次数")
    parser.add_argument("--seeds", type=str, default="74", help="用于每次尝试的逗号分隔的种子列表 (例如 '321,105,421')")
    parser.add_argument("--seed", type=int, default=42, help="vLLM 环境的全局初始化种子")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p 采样")
    parser.add_argument("--max_tokens", type=int, default=14000, help="要生成的最大 token 数")
    
    args = parser.parse_args()
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
        
    main(args)
