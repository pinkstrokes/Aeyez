#!/usr/bin/env python3
"""
Simple LLaVA-1.5-7B inference on MMMU using Transformers
No vLLM, no multi-agent flow - just direct model inference
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import string
from concurrent.futures import ThreadPoolExecutor

# Local imports
from dataset_utils import load_dataset, dump_image, MMMU_preproc
from eval_utils import build_judge, eval_single_sample


class LLaVAModel:
    """Simple LLaVA model wrapper using Transformers"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"Loading LLaVA model from {model_path}")
        print(f"Using device: {device}")

        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.dump_image_func = None

        print("Model loaded successfully!")

    def set_dump_image(self, func):
        self.dump_image_func = func

    def build_prompt(self, sample, dataset_name):
        """Build prompt from MMMU sample"""
        question = sample['question']
        options_prompt = ""

        # Build options in MMMU standard format
        for ch in ['A', 'B', 'C', 'D']:
            if ch in sample and not pd.isna(sample[ch]):
                options_prompt += f"{ch}. {sample[ch]}\n"

        # Build full prompt following MMMU format
        if options_prompt:
            question_text = f"Question: {question}\nOptions:\n{options_prompt}Please select the correct answer from the options above."
        else:
            question_text = f"Question: {question}"

        # Handle images
        image_paths = []
        if 'image' in sample:
            image_val = sample['image']
            has_image = False

            if isinstance(image_val, (list, np.ndarray)):
                has_image = len(image_val) > 0 and not all(pd.isna(img) for img in image_val)
            else:
                has_image = not pd.isna(image_val)

            if has_image and self.dump_image_func is not None:
                image_paths = self.dump_image_func(sample)

        return question_text, image_paths

    def generate(self, text, image_paths, max_new_tokens=512, temperature=0.01):
        """Generate response from text and images"""
        # Load images
        images = []
        if image_paths:
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")

        # Prepare conversation format
        if images:
            # Add one image token per image
            content = []
            for _ in images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": text})

            conversation = [
                {
                    "role": "user",
                    "content": content,
                },
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                },
            ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            images=images if images else None,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode
        generated_text = self.processor.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()


def run_inference(args):
    """Run inference on MMMU dataset"""
    print("="*60)
    print("LLaVA-1.5-7B MMMU Inference")
    print("="*60)

    # Load dataset
    data = load_dataset(args.dataset)
    print(f"Loaded dataset: {len(data)} samples")

    # Filter to validation split only if requested
    if args.split:
        data = data[data['split'] == args.split]
        print(f"Filtered to {args.split} split: {len(data)} samples")

    # Set up image directory
    img_root = os.path.join(args.data_dir, 'images', 'MMMU')
    os.makedirs(img_root, exist_ok=True)

    def dump_image_func(line):
        return dump_image(line, img_root)

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Set up CoT prompt if enabled
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        print(f"Using CoT prompt: {cot_prompt}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = LLaVAModel(args.model_path, device=device)
    model.set_dump_image(dump_image_func)

    # Handle sample selection
    start_idx = args.start_sample if args.start_sample else 0
    end_idx = args.end_sample if args.end_sample else None
    max_samples = args.max_samples

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
    failed_samples = []
    for i in tqdm(range(len(data)), desc="Running inference"):
        line = data.iloc[i]
        index = line['index']

        try:
            # Convert to dict
            line_dict = line.to_dict()
            for k, v in line_dict.items():
                if isinstance(v, np.integer):
                    line_dict[k] = int(v)
                elif isinstance(v, np.floating):
                    line_dict[k] = float(v)

            # Build prompt
            text, image_paths = model.build_prompt(line, args.dataset)

            # Add CoT prompt if enabled
            if args.use_cot:
                text += cot_prompt

            # Generate response
            response = model.generate(
                text,
                image_paths,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )

            print(f"\nSample {i}:")
            print(f"Question: {text[:100]}...")
            print(f"Response: {response}")
            print(f"Expected: {line['answer']}")
            print('-' * 50)

            # Save result
            result = {
                "question_id": int(index) if isinstance(index, np.integer) else index,
                "annotation": line_dict,
                "task": args.dataset,
                "result": {"gen": response}
            }
            results.append(result)

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error processing sample {i} (index={index}): {error_msg}")

            # Log more details for image-related errors
            if "Image features and image tokens do not match" in error_msg:
                img_count = len(image_paths) if 'image_paths' in locals() else 0
                print(f"   Images in sample: {img_count}")
                if img_count > 0:
                    print(f"   Image paths: {image_paths[:3]}...")  # Show first 3

            failed_samples.append((i, index, error_msg))
            # Continue to next sample
            continue

        # Save intermediate results every 10 samples
        if (i + 1) % 10 == 0:
            with open(args.output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')

    # Write final results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

    print(f"\n✅ Inference completed! Results saved to {args.output_file}")
    print(f"Successfully processed: {len(results)} samples")
    if failed_samples:
        print(f"⚠️ Failed samples: {len(failed_samples)}")
        for i, idx, err in failed_samples[:5]:  # Show first 5
            print(f"  - Sample {i} (index={idx}): {err[:100]}")


def run_evaluation(args):
    """Run evaluation on inference results"""
    print("="*60)
    print("MMMU Evaluation")
    print("="*60)

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

    # Normalize column names
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    # Load ground truth
    meta = load_dataset(args.dataset)

    print(f"Loaded {len(data)} predictions")
    print(f"Loaded {len(meta)} ground truth samples")

    # Validate
    meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
    data_map = {x: y for x, y in zip(data['index'], data['question'])}

    for k in data_map:
        assert k in meta_q_map, f'Sample {k} not found in ground truth'

    # Prepare data
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    data = MMMU_preproc(data)
    answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}
    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]

    items = []
    for i in range(len(data)):
        items.append(data.iloc[i])

    # Build judge model
    model = build_judge(args.eval_model, args.api_type)

    # Run evaluation
    eval_tasks = [(model, item) for item in items]
    eval_results = []

    debug = os.environ.get('DEBUG', '').lower() == 'true'
    if debug:
        print("Debug mode: evaluating first 5 samples")
        for task in eval_tasks[:5]:
            result = eval_single_sample(task)
            eval_results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.nproc) as executor:
            for result in tqdm(executor.map(eval_single_sample, eval_tasks),
                             total=len(eval_tasks), desc="Evaluating"):
                eval_results.append(result)

    # Calculate accuracy
    accuracy = sum(r['hit'] for r in eval_results) / len(eval_results)

    # Calculate by split
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
        print(f"Accuracy for {split}: {split_accuracy:.4f} ({sum(r['hit'] for r in split_results)}/{len(split_results)})")

    # Save results
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)

    # Save accuracy
    acc_file = args.output_file.replace('.csv', '_acc.json')
    with open(acc_file, 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "accuracy_by_split": accuracy_by_split
        }, f, indent=2)

    print(f"\n✅ Overall Accuracy: {accuracy:.4f}")
    print(f"Results saved to {args.output_file}")
    print(f"Accuracy metrics saved to {acc_file}")


def main():
    parser = argparse.ArgumentParser(description="Simple LLaVA-1.5-7B MMMU Evaluation")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str,
                             default="/projects/bdpn/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
                             help="Path to LLaVA model")
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL",
                             help="Dataset name")
    infer_parser.add_argument("--data-dir", type=str,
                             default="/u/hli36/MPU-RL-clean/src/multi-agent/benchmark_evaluation/llava/inputs",
                             help="Data directory")
    infer_parser.add_argument("--output-file", type=str,
                             default="outputs/inference_results.jsonl",
                             help="Output file")
    infer_parser.add_argument("--max-samples", type=int, help="Max samples to process")
    infer_parser.add_argument("--split", type=str, choices=['dev', 'validation'],
                             help="Filter to specific split (dev or validation)")
    infer_parser.add_argument("--start-sample", type=int, help="Start sample index")
    infer_parser.add_argument("--end-sample", type=int, help="End sample index")
    infer_parser.add_argument("--max-tokens", type=int, default=512,
                             help="Max tokens to generate")
    infer_parser.add_argument("--temperature", type=float, default=0.01,
                             help="Sampling temperature")
    infer_parser.add_argument("--cpu", action="store_true",
                             help="Force CPU usage (default: use GPU if available)")
    infer_parser.add_argument("--use-cot", action="store_true",
                             help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="",
                             help="Custom Chain-of-Thought prompt")

    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str,
                            default="/u/hli36/MPU-RL-clean/src/multi-agent/benchmark_evaluation/llava/inputs",
                            help="Data directory")
    eval_parser.add_argument("--input-file", type=str,
                            default="outputs/inference_results.jsonl",
                            help="Input inference results")
    eval_parser.add_argument("--output-file", type=str,
                            default="outputs/eval_results.csv",
                            help="Output file")
    eval_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL",
                            help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="qwen-flash",
                            choices=["gpt-3.5-turbo-0125","gpt-4-0125-preview","qwen-flash","qwen-plus"],
                            help="Model for evaluation")
    eval_parser.add_argument("--api-type", type=str, default="dash",
                            choices=["dash", "mit"],
                            help="API type for evaluation")
    eval_parser.add_argument("--nproc", type=int, default=4,
                            help="Number of processes")

    args = parser.parse_args()

    # Set environment
    os.environ['LMUData'] = args.data_dir

    if args.mode == "infer":
        run_inference(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
