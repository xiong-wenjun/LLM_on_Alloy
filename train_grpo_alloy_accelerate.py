import argparse
import json
import torch
import tqdm
import os
import re
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
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""

    final_answer = final_answer.split("=")[-1]
    # Replace irrelevant content
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Remove parentheses related to \left and \right
    final_answer = re.sub(r"\\left\s*\(", "(", final_answer)
    final_answer = re.sub(r"\\right\s*\)", ")", final_answer)
    final_answer = re.sub(r"\\left\s*\[", "[", final_answer)
    final_answer = re.sub(r"\\right\s*\]", "]", final_answer)
    final_answer = re.sub(r"\\left\s*\\\{", "{", final_answer)
    final_answer = re.sub(r"\\right\s*\\\}", "}", final_answer)
    # Just in case there are standalone \left or \right
    final_answer = final_answer.replace(r"\left", "")
    final_answer = final_answer.replace(r"\right", "")

    # Extract the latex math part
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def verify_answer_predict(generated_answer, final_ground_truth):
    '''predict approxiamate value of accurate answer'''
    final_answer = extract_answer(generated_answer)
    if final_answer == "NA":
        return -1.0, False, final_answer  # r acc pred
    final_answer = normalize_final_answer(final_answer)
    temp = final_answer
    try:
        final_answer = extract_numeric_value(final_answer)
    except ValueError:
        return -1.0, False, temp

    # Calculate error percentage (absolute value)
    error_percentage = (abs(final_answer - final_ground_truth) / (abs(final_ground_truth) + 1e-7)) * 100

    # Check if within 20% range
    is_correct = error_percentage <= 20

    # Calculate reward value: linearly changes from 0 to 1 within the 20% range
    # Reward is 1 for a perfect match, and 0 at the 20% boundary
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
        return after_boxed[start_index + 1:end_index]
    else:
        return "NA"


def extract_numeric_value(input_str):
    match = re.search(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', input_str)

    if not match:
        raise ValueError("No numeric part found in the input string")

    numeric_str = match.group(1)
    numeric_value = float(numeric_str)

    if '%' in input_str:
        return numeric_value / 100.0
    else:
        return numeric_value


def main(args):
    """
    Perform batch inference and pass@k evaluation using vLLM.
    This script iterates up to k times (using different seeds) and removes successfully solved problems from subsequent runs.
    """
    model_name = os.path.basename(os.path.normpath(args.model_path))
    print("=" * 20 + f" Starting Pass@{args.k} evaluation for model: {model_name} " + "=" * 20)

    # 1. Parse seed list
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        if len(seeds) < args.k:
            raise ValueError(f"The number of provided seeds ({len(seeds)}) is less than k ({args.k}). Please provide at least k seeds.")
    except Exception as e:
        print(f"Error parsing seeds: {e}")
        return

    # 2. Load model and tokenizer
    print(f"Loading model and tokenizer from {args.model_path}...")
    try:
        # Note: For reproducibility, the seed parameter here in vLLM is used to initialize the distributed environment.
        # We will pass different seeds via sampling_params in each loop to vary the generation results.
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            seed=args.seed,  # Base initialization seed
            device="cuda" 
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        return

    # 3. Load dataset
    print(f"Loading JSON data from {args.input_file}...")
    try:
        dataset = Dataset.from_json(args.input_file)
        total_count = len(dataset)
        print(f"Successfully loaded {total_count} records.")
    except Exception as e:
        print(f"Failed to load JSON file: {e}")
        return

    # 4. Initialize result tracking
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

    # 5. Pass@k loop
    for attempt in range(1, args.k + 1):
        current_seed = seeds[attempt - 1]
        print(f"\n--- Attempt {attempt}/{args.k} | Using seed: {current_seed} ---")

        if not unsolved_ids:
            print("🎉 All problems have been solved! Terminating evaluation early.")
            break

        unsolved_dataset = [d for d in dataset if d["id"] in unsolved_ids]
        print(f"Number of problems to solve: {len(unsolved_dataset)}")

        # Configure sampling parameters
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            skip_special_tokens=True,
            seed=current_seed,  # Set seed for this attempt
        )

        newly_solved_in_this_attempt = 0

        # Batch processing loop
        for i in tqdm.tqdm(range(0, len(unsolved_dataset), batch_size), desc=f"Processing attempt {attempt}"):
            batch_data = unsolved_dataset[i:i + batch_size]

            batch_prompts, batch_ids, batch_ground_truths = [], [], []
            for item in batch_data:
                prompt_template = r"""You are a materials science expert. Please answer the following question, and put the final answer inside the LaTeX symbol \boxed{}. The output inside \boxed{} must adhere to the following unit conventions:
- **Yield strength, tensile strength, or similar:** Numerical value (without units) if the answer is in MPa.
- **Elongation or similar:** Percentage value (with the '%' symbol).
- **Hardness or similar:** Numerical value (without units) if the answer is in HV.

Do not include units such as 'MPa', '%', 'HV', or any other text in your response. Only provide the true value as specified in the ground truth. For example:
- If the ground truth is '120 MPa', output: \boxed{120}
- If the ground truth is '17.0%', output: \boxed{17.0%}
- If the ground truth is '200 HV', output: \boxed{200}
For the remaining data types, please use common units.

Here's the format request:
Please provide your reasoning process enclosed within <think> and </think> tags, followed by your final answer with explanation. i.e., <think> your reasoning here </think> your answer here, including \boxed{final answer}
Please ensure your output is as concise as possible while maintaining accuracy and reliability. When answering, beyond recalling and remembering inherent material properties, focus more on the application of these properties and logical scientific derivation.
Let's think step by step.
"""
                batch_prompts.append(prompt_template + "\n\n" + item["question"])
                batch_ids.append(item["id"])
                batch_ground_truths.append(item["ideal_answer"])

            # Skip empty batches
            if not batch_prompts:
                continue

            # Generate answers
            batch_outputs = llm.generate(batch_prompts, sampling_params)

            for j, request_output in enumerate(batch_outputs):
                item_id = batch_ids[j]
                ground_truth = batch_ground_truths[j]
                generated_answer = request_output.outputs[0].text.strip()

                try:
                    score, is_correct, model_prediction = verify_answer_predict(generated_answer, float(re.search(r'([+-]?\d+\.?\d*)', ground_truth).group(1)))
                except Exception as e:
                    print(f"\nWarning: Error calling verify_answer_predict for ID {item_id}: {e}")
                    is_correct = False
                    model_prediction = f"EVALUATION_ERROR: {e}"

                # Record generation result
                generation_data = {
                    "attempt": attempt,
                    "seed": current_seed,
                    "generated_answer": generated_answer,
                    "model_prediction": model_prediction,
                    "is_correct": bool(is_correct)
                }
                all_results[item_id]["generations"].append(generation_data)

                # Update solved status
                if is_correct and not all_results[item_id]["is_correct_in_k"]:
                    all_results[item_id]["is_correct_in_k"] = True
                    all_results[item_id]["solved_at_attempt"] = attempt
                    unsolved_ids.remove(item_id)
                    correct_in_k_count += 1
                    newly_solved_in_this_attempt += 1

        print(f"Attempt {attempt} completed. Newly solved problems: {newly_solved_in_this_attempt}")
        current_accuracy = correct_in_k_count / len(dataset)
        print(f"Current Pass@{args.k} accuracy: {current_accuracy:.4f} ({correct_in_k_count}/{len(dataset)})")

        # 6. Save intermediate results
        # Derive intermediate filename from the final output filename
        output_dir = os.path.dirname(args.output_file)
        base_name = os.path.basename(args.output_file).replace('.json', '')
        intermediate_output_path = os.path.join(output_dir, f"{base_name}_attempt_{attempt}.json")

        print(f"Saving intermediate results for attempt {attempt} to {intermediate_output_path}...")
        try:
            # Create a copy for the intermediate output to avoid modifying data structures in the main loop
            intermediate_output = {
                "model_name": model_name,
                "pass_at_k_configuration": args.k,
                "current_attempt": attempt,
                "current_seed": current_seed,
                "current_pass_rate": current_accuracy,
                "total_samples": total_count,
                "correct_samples_so_far": correct_in_k_count,
                "results": list(all_results.values())  # Convert to list for serialization
            }
            with open(intermediate_output_path, 'w', encoding='utf-8') as f:
                json.dump(intermediate_output, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Failed to save intermediate results file: {e}")

    # 7. Calculate final accuracy and prepare final output file
    final_accuracy = (correct_in_k_count / total_count) if total_count > 0 else 0
    print("\n" + "=" * 20 + " Evaluation Complete " + "=" * 20)
    print(f"Model: {model_name}")
    print(f"Total samples: {total_count}")
    print(f"Pass@{args.k} correct count: {correct_in_k_count}")
    print(f"Final Pass@{args.k} accuracy: {final_accuracy:.4f}")

    final_output = {
        "model_name": model_name,
        "pass_at_k": args.k,
        "accuracy": final_accuracy,
        "total_samples": total_count,
        "correct_samples": correct_in_k_count,
        "results": list(all_results.values())  # Convert to list for serialization
    }

    # 8. Save final results to JSON file
    print(f"Saving final evaluation results to {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        print(f"All tasks completed! Evaluation results successfully saved to {args.output_file}")
    except Exception as e:
        print(f"Failed to save final results file: {e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Perform efficient pass@k evaluation on language models using vLLM.")
    # Path parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="/gemini/code/grpo/Qwen2.5-7B-Instruct-GRPO_step86110",
        help="Path to the base LLM model to be evaluated"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/mnt-nfsdata/zhengye/qna_dataset_pipeline/setting1/final_output/0723_val_dataset_400.json",
        help="Path to the input .json file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/gemini/code/outputs/baseline_results_step10.json",
        help="Path to the final output .json results file"
    )

    # vLLM performance parameters
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization for vLLM")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")

    # Model generation and evaluation parameters
    parser.add_argument("--k", "--pass_at_k", type=int, default=1, help="Maximum number of attempts per problem in pass@k evaluation")
    parser.add_argument("--seeds", type=str, default="74", help="Comma-separated list of seeds for each attempt (e.g., '321,105,421')")
    parser.add_argument("--seed", type=int, default=42, help="Global initialization seed for the vLLM environment")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=14000, help="Maximum number of tokens to generate")

    args = parser.parse_args()
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    main(args)