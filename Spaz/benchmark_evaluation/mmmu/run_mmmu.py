import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor
import string
import traceback

# Local imports from refactored files
from dataset_utils import load_dataset, dump_image, MMMU_preproc
from eval_utils import build_judge, eval_single_sample

from qwen2_vl.model import Qwen2VLChat
from qwen_vl_utils import process_vision_info

def run_inference(args):
    """Run inference on the MMMU dataset."""
    # Load dataset
    data = load_dataset(args.dataset)
    
    # Set up image root directory
    img_root = os.path.join(os.environ['LMUData'], 'images', 'MMMU')
    os.makedirs(img_root, exist_ok=True)
    
    # Set up dump_image function
    def dump_image_func(line):
        return dump_image(line, img_root)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Set up CoT prompt if enabled
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        print(f"Using CoT prompt: {cot_prompt}")

    # Initialize model using existing app/llm.py factory
    if args.api_type == 'app_llm':
        # Use existing LLM factory from app/llm.py
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
        from app.llm import LLM
        print(f"Using app/llm.py factory with config: {args.config_name}")

        # Create a wrapper to make LLM compatible with official evaluator
        class AppLLMWrapper:
            def __init__(self, config_name):
                self.llm = LLM(config_name)
                self.config_name = config_name
                self.dump_image_func = None

                # Fix DashScope base_url if it's empty
                if hasattr(self.llm, 'provider') and hasattr(self.llm.provider, 'api_base'):
                    if not self.llm.provider.api_base:
                        self.llm.provider.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
                        print(f"Fixed DashScope base_url: {self.llm.provider.api_base}")

            def set_dump_image(self, func):
                self.dump_image_func = func

            def build_prompt(self, sample, dataset_name):
                # Convert MMMU sample to app/llm format
                question = sample['question']
                choices = []
                for ch in ['A', 'B', 'C', 'D']:
                    if ch in sample and not pd.isna(sample[ch]):
                        choices.append(f"({ch}) {sample[ch]}")

                if choices:
                    question_text = f"{question}\n\n" + "\n".join(choices)
                else:
                    question_text = question

                # Handle images if present
                messages = []
                if 'image' in sample:
                    image_val = sample['image']
                    # Handle both single images and multiple images
                    has_image = False
                    if isinstance(image_val, (list, np.ndarray)):
                        # Multiple images - check if any are valid
                        has_image = len(image_val) > 0 and not all(pd.isna(img) for img in image_val)
                    else:
                        # Single image - check if it's valid
                        has_image = not pd.isna(image_val)

                    if has_image:
                        # Dump image to file
                        if self.dump_image_func is not None:
                            image_paths = self.dump_image_func(sample)
                            # Add image message (format depends on your LLM provider)
                            messages.append({"type": "image", "value": image_paths[0] if image_paths else None})
                        else:
                            print("Warning: dump_image_func not set, skipping image processing")

                messages.append({"type": "text", "value": question_text})
                return messages

            def generate(self, messages):
                # Convert messages to app/llm format and call
                import asyncio

                # Convert to simple text for now (will need multimodal support)
                text_content = ""
                for msg in messages:
                    if msg["type"] == "text":
                        text_content += msg["value"]

                # Use asyncio to call the async LLM
                result = asyncio.run(self.llm.ask([{"role": "user", "content": text_content}]))
                return result

        model = AppLLMWrapper(args.config_name)
        model.set_dump_image(dump_image_func)

    elif args.api_type == 'local':
        # Use local HuggingFace model (original code)
        print(f"Loading local HuggingFace model from {args.model_path}")
        model = Qwen2VLChat(
            model_path=args.model_path,
            temperature=0.01,
            top_p=0.001,
            top_k=1,
            use_custom_prompt=True,
            min_pixels=1280*28*28,
            max_pixels=5120*28*28
        )
        model.set_dump_image(dump_image_func)
    else:
        raise ValueError(f"Unsupported API type: {args.api_type}")

    # Handle sample range selection
    start_idx = getattr(args, 'start_sample', 0) or 0
    end_idx = getattr(args, 'end_sample', None)
    max_samples = getattr(args, 'max_samples', None)

    print(f"Debug: start_idx={start_idx}, end_idx={end_idx}, max_samples={max_samples}")
    print(f"Debug: Total dataset size: {len(data)}")

    if start_idx > 0 or end_idx is not None:
        if end_idx is None:
            end_idx = len(data)
        data = data.iloc[start_idx:end_idx]
        print(f"Processing samples {start_idx} to {end_idx-1} ({len(data)} samples)")
    elif max_samples and max_samples > 0:
        data = data.iloc[:max_samples]
        print(f"Limited to {max_samples} samples")
    else:
        print(f"Processing all {len(data)} samples")

    # Run inference
    results = []
    for i in tqdm(range(len(data)), desc="Running inference"):
        # line = data.iloc[i].to_dict()
        line = data.iloc[i]
        index = line['index']
        
        # Convert line to dict and ensure all values are JSON serializable
        line_dict = line.to_dict()
        for k, v in line_dict.items():
            if isinstance(v, np.integer):
                line_dict[k] = int(v)
            elif isinstance(v, np.floating):
                line_dict[k] = float(v)
        
        # Generate response using HuggingFace
        messages = model.build_prompt(line, args.dataset)
        
        # Add CoT prompt if enabled
        if args.use_cot and len(messages) > 0 and messages[-1]['type'] == 'text':
            messages[-1]['value'] += cot_prompt
            
        response = model.generate(messages)
            
        print(f"response: {response}")
        print(f"annotation answer: {line['answer']}")
        print('-' * 50)
        
        # Save result
        result = {
            "question_id": int(index) if isinstance(index, np.integer) else index,
            "annotation": line_dict,
            "task": args.dataset,
            "result": {"gen": response},
            "messages": messages
        }
        results.append(result)
        
        # Write intermediate results
        if i % 10 == 0:
            with open(args.output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')
            
    # Write final results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"Inference completed. Results saved to {args.output_file}")

def run_evaluation(args):
    """Run evaluation on inference results."""
    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            annotation["prediction"] = job["result"]["gen"]
            results.append(annotation)
            
    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    # If not choice label, then use lower case
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    # Load dataset
    meta = load_dataset(args.dataset)

    # data中data.iloc[i]中的index必须在results中存在，对应results中的results[i]['id']，并且data中data.iloc[i]中的question必须和results中results[i]['annotation']中的question完全一致
    print(f"len(data): {len(data)}")
    print(f"len(meta): {len(meta)}")
    meta_q_map = {str(x): y for x, y in zip(meta['index'], meta['question'])}
    data_map = {str(x): y for x, y in zip(data['index'], data['question'])}
    for k in data_map:
        assert k in meta_q_map, (
            f'eval_file should be the same as or a subset of dataset MMMU_DEV_VAL'
        )

    answer_map = {str(i): c for i, c in zip(meta['index'], meta['answer'])}
    data = MMMU_preproc(data)
    answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}
    # Ensure index is string for consistent lookup
    data['index'] = data['index'].astype(str)
    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    items = []
    for i in range(len(data)):
        item = data.iloc[i]
        items.append(item)

    # Build judge model if needed
    model = None
    model = build_judge(args.eval_model, args.api_type)
    
    # Prepare evaluation tasks
    eval_tasks = []
    for item in items:
        eval_tasks.append((model, item))
    
    # Run evaluation
    eval_results = []
    
    # Debug mode: process single-threaded with first few samples
    debug = os.environ.get('DEBUG', '').lower() == 'true'
    if debug:
        print("Running in debug mode with first 5 samples...")
        # for task in tqdm(eval_tasks[:5], desc="Evaluating"):
        for task in eval_tasks[:5]:
            try:
                result = eval_single_sample(task)
                eval_results.append(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                print(f"Task details: {task}")
                raise
    else:
        # Normal mode: process all samples with threading
        nproc = getattr(args, 'nproc', 4)  # Default to 4 if not set
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            for result in tqdm(executor.map(eval_single_sample, eval_tasks), 
                             total=len(eval_tasks), desc="Evaluating"):
                eval_results.append(result)
    
    # Calculate overall accuracy
    accuracy = sum(r['hit'] for r in eval_results) / len(eval_results)
    
    # Calculate accuracy by split
    results_by_split = {}
    for result in eval_results:
        split = result.get('split', 'unknown')
        if split not in results_by_split:
            results_by_split[split] = []
        results_by_split[split].append(result)
    
    accuracy_by_split = {}
    for split, split_results in results_by_split.items():
        split_accuracy = sum(r['hit'] for r in split_results) / len(split_results)
        accuracy_by_split[split] = split_accuracy
        print(f"Accuracy for {split} split: {split_accuracy:.4f} ({sum(r['hit'] for r in split_results)}/{len(split_results)})")
    
    # Save results
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)
    
    # Save accuracy
    with open(args.output_file.replace('.csv', '_acc.json'), 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "accuracy_by_split": accuracy_by_split
        }, f, indent=2)
    
    # print(f"Evaluation completed. Overall accuracy: {accuracy:.4f}")
    print(f"Results saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="MMMU Evaluation Script")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--api-type", type=str, default="app_llm",
                             choices=["app_llm", "local"],
                             help="API type: app_llm (use app/llm.py factory) or local (use HuggingFace)")
    infer_parser.add_argument("--config-name", type=str, default="qwen2_5_vl_3b_dashscope",
                             help="Config name from config.toml (when using app_llm)")
    infer_parser.add_argument("--model-path", type=str, help="Path to local model (when using local)")
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    infer_parser.add_argument("--data-dir", type=str,
                             default="/projects/bdpn/wzhang29/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/inputs",
                             help="The absolute path of MMMU_DEV_VAL.tsv")
    infer_parser.add_argument("--output-file", type=str,
                             default="/projects/bdpn/wzhang29/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/outputs/official_inference_results.jsonl",
                             help="Output file path")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt")
    infer_parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    infer_parser.add_argument("--start-sample", type=int, help="Starting sample index (0-based)")
    infer_parser.add_argument("--end-sample", type=int, help="Ending sample index (exclusive)")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str,
                            default="/projects/bdpn/wzhang29/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/inputs",
                            help="The absolute path of MMMU_DEV_VAL.tsv")
    eval_parser.add_argument("--input-file", type=str,
                            default="/projects/bdpn/wzhang29/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/outputs/official_inference_results.jsonl",
                            help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str,
                            default="/projects/bdpn/wzhang29/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/outputs/official_eval_results.csv",
                            help="Output file path")
    eval_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="qwen-flash",
                            choices=["gpt-3.5-turbo-0125","gpt-4-0125-preview","qwen-flash","qwen-plus"],
                            help="Model to use for evaluation")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"],
                            help="API type to use for evaluation")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use")
    
    args = parser.parse_args()

    os.environ['LMUData'] = args.data_dir
    
    if args.mode == "infer":
        run_inference(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
