#!/usr/bin/env python3
"""
LLaVA-1.5-7B inference on MMMU-Pro using local model
Adapted from llava/run_mmmu.py for MMMU-Pro standard (10 options) subset
"""

import os
import sys
import json
import argparse
import re
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from typing import List, Dict, Any, Tuple


class LLaVAModel:
    """LLaVA model wrapper for MMMU-Pro"""

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
        print("Model loaded successfully!")

    def generate(self, text, images, max_new_tokens=512, temperature=0.01):
        """Generate response from text and images"""
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

        # Prepare inputs
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


class MMMUProPromptManager:
    """Manages MMMU-Pro prompts (adapted from mmmu_pro/prompt_utils.py)"""

    OFFICIAL_PROMPTS = {
        "cot": {
            "standard": "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
        },
        "direct": {
            "standard": "Answer with the option letter from the given choices directly."
        }
    }

    def replace_image_tokens(self, input_string: str) -> Tuple[str, List[int]]:
        """Replace <image i> tokens and record their original order"""
        pattern = r'<image (\d+)>'
        matches = re.findall(pattern, input_string)
        image_order = [int(match) for match in matches]
        processed_string = re.sub(pattern, '[IMAGE]', input_string)
        return processed_string, image_order

    def process_options_with_images(self, options: List[str], images: List[Image.Image]) -> Tuple[str, List[Image.Image]]:
        """Process options that may contain image tokens and reorder images accordingly"""
        if not options:
            return "", images

        # Create image mapping (1-indexed to 0-indexed)
        image_map = {i+1: img for i, img in enumerate(images)}

        formatted_options = []
        used_images = []

        for i, option in enumerate(options):
            letter = chr(ord('A') + i)

            # Check if option contains image token
            if '<image' in option:
                processed_option, image_order = self.replace_image_tokens(option)

                # Add images in the order they appear in this option
                for img_idx in image_order:
                    if img_idx in image_map:
                        used_images.append(image_map[img_idx])

                formatted_options.append(f"{letter}. {processed_option}")
            else:
                formatted_options.append(f"{letter}. {option}")

        # If no images were used in options, return original images
        if not used_images:
            used_images = images

        options_text = "\n".join(formatted_options)
        return options_text, used_images

    def build_prompt(self, sample: Dict[str, Any], mode: str = "cot") -> Tuple[str, List[Image.Image]]:
        """Build prompt for MMMU-Pro standard (10 options) inference"""
        question = sample['question']
        options = sample.get('options', [])
        images = sample.get('images', [])

        # Process options and handle image token reordering
        options_text, reordered_images = self.process_options_with_images(options, images)

        # Build full prompt
        if options_text:
            full_prompt = f"{question}\n\n{options_text}\n\n{self.OFFICIAL_PROMPTS[mode]['standard']}"
        else:
            full_prompt = f"{question}\n\n{self.OFFICIAL_PROMPTS[mode]['standard']}"

        return full_prompt, reordered_images


def extract_answer_from_response(response: str, options: List[str]) -> str:
    """Extract answer from model response using official MMMU-Pro logic"""
    response = response.strip()

    # First, look for the official format: "Answer: X"
    official_pattern = r'Answer:\s*([A-J])'
    matches = re.findall(official_pattern, response.upper(), re.IGNORECASE)
    if matches:
        answer = matches[-1].upper()
        # Validate answer is within valid range
        if ord(answer) - ord('A') < len(options):
            return answer

    # If no official format found, try other patterns
    patterns = [
        r'(?:final answer|the answer|my answer)(?:\s*is)?\s*:?\s*([A-J])',
        r'(?:select|choose|pick)\s*(?:option\s*)?([A-J])',
        r'(?:^|\n)\s*([A-J])\s*[.)]?\s*$',
        r'\b([A-J])\b(?=\s*$)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response.upper(), re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1].upper()
            if ord(answer) - ord('A') < len(options):
                return answer

    # Last resort: find any single letter in the response
    letters = re.findall(r'\b([A-J])\b', response.upper())
    if letters:
        answer = letters[-1]
        if ord(answer) - ord('A') < len(options):
            return answer

    return response


def run_inference(args):
    """Run inference on MMMU-Pro standard (10 options) validation set"""
    print("="*60)
    print("LLaVA-1.5-7B MMMU-Pro Standard (10 options) Inference")
    print("="*60)

    # Load dataset
    print(f"Loading MMMU-Pro standard (10 options) dataset...")
    dataset = load_dataset("MMMU/MMMU_Pro", "standard (10 options)")

    # Filter to validation samples only
    all_samples = list(dataset['test'])
    validation_samples = [s for s in all_samples if s.get('id', '').startswith('validation_')]
    print(f"Total test samples: {len(all_samples)}")
    print(f"Validation samples: {len(validation_samples)}")

    # Limit samples if requested
    if args.max_samples and args.max_samples > 0:
        validation_samples = validation_samples[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load model
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = LLaVAModel(args.model_path, device=device)

    # Initialize prompt manager
    prompt_manager = MMMUProPromptManager()

    # Run inference
    results = []
    failed_samples = []

    for i in tqdm(range(len(validation_samples)), desc="Running inference"):
        sample = validation_samples[i]
        sample_id = sample.get('id', f'sample_{i}')

        try:
            # Process sample and parse options
            options = sample['options']
            if isinstance(options, str):
                # Parse string representation of list
                import ast
                try:
                    options = ast.literal_eval(options)
                except (ValueError, SyntaxError):
                    options = [options]

            processed_sample = {
                'question': sample['question'],
                'options': options,
                'images': [sample['image']] if sample.get('image') else [],
                'answer': sample['answer']
            }

            # Build prompt
            prompt_text, images = prompt_manager.build_prompt(processed_sample, mode=args.mode)

            # Generate response
            response = model.generate(
                prompt_text,
                images,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )

            # Extract answer
            extracted_answer = extract_answer_from_response(response, processed_sample['options'])

            print(f"\nSample {i} (ID: {sample_id}):")
            print(f"Question: {sample['question'][:80]}...")
            print(f"Response: {response[:150]}...")
            print(f"Extracted: {extracted_answer}")
            print(f"Ground Truth: {sample['answer']}")
            print('-' * 50)

            # Save result
            result = {
                "id": sample_id,
                "subset": "standard (10 options)",
                "subject": sample.get('subject', ''),
                "topic_difficulty": sample.get('topic_difficulty', ''),
                "question": sample['question'],
                "options": options,  # Use parsed options
                "ground_truth": sample['answer'],
                "model_response": response,
                "extracted_answer": extracted_answer
            }
            results.append(result)

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error processing sample {i} (ID={sample_id}): {error_msg}")
            failed_samples.append((i, sample_id, error_msg))

            # Save error result
            result = {
                "id": sample_id,
                "subset": "standard (10 options)",
                "subject": sample.get('subject', ''),
                "error": error_msg
            }
            results.append(result)
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
    print(f"Successfully processed: {len(results) - len(failed_samples)} samples")
    if failed_samples:
        print(f"⚠️ Failed samples: {len(failed_samples)}")
        for i, sid, err in failed_samples[:5]:
            print(f"  - Sample {i} (ID={sid}): {err[:100]}")


def run_evaluation(args):
    """Run evaluation on inference results"""
    print("="*60)
    print("MMMU-Pro Evaluation")
    print("="*60)

    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))

    print(f"Loaded {len(results)} results")

    # Evaluate
    total = 0
    correct = 0
    errors = 0

    evaluations = []

    for result in results:
        if 'error' in result:
            errors += 1
            continue

        total += 1
        ground_truth = result['ground_truth'].strip().upper()
        predicted = result['extracted_answer'].strip().upper()

        is_correct = (ground_truth == predicted)
        if is_correct:
            correct += 1

        evaluations.append({
            'id': result['id'],
            'subject': result.get('subject', ''),
            'correct': is_correct,
            'ground_truth': ground_truth,
            'predicted': predicted
        })

    accuracy = correct / total if total > 0 else 0

    # Save evaluation results
    import pandas as pd
    df = pd.DataFrame(evaluations)
    df.to_csv(args.output_file, index=False)

    # Save accuracy metrics
    acc_file = args.output_file.replace('.csv', '_acc.json')
    with open(acc_file, 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors
        }, f, indent=2)

    print(f"\n✅ Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Errors: {errors}")
    print(f"Results saved to {args.output_file}")
    print(f"Accuracy metrics saved to {acc_file}")


def main():
    parser = argparse.ArgumentParser(description="LLaVA-1.5-7B MMMU-Pro Evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str,
                             default="/projects/bdpn/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
                             help="Path to LLaVA model")
    infer_parser.add_argument("--output-file", type=str,
                             default="outputs/mmmu_pro/inference_results.jsonl",
                             help="Output file")
    infer_parser.add_argument("--max-samples", type=int, help="Max samples to process")
    # MMMU Official CoT parameters reference:
    # - Original MMMU: max_new_tokens=128, temperature=1.0, do_sample=True, num_beams=5
    # - MMMU Pro (lmdeploy): max_new_tokens=4096, temperature=0.8, top_p=0.95
    # For CoT mode, we need more tokens than 128 to allow reasoning
    infer_parser.add_argument("--max-tokens", type=int, default=1024,
                             help="Max tokens to generate (official: 128 for direct, 1024+ for CoT)")
    infer_parser.add_argument("--temperature", type=float, default=0.7,
                             help="Sampling temperature (official: 1.0, lower for more deterministic)")
    infer_parser.add_argument("--cpu", action="store_true",
                             help="Use CPU instead of GPU")
    infer_parser.add_argument("--mode", type=str, default="cot", choices=["cot", "direct"],
                             help="Prompt mode: 'cot' (chain-of-thought, recommended) or 'direct'")

    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--input-file", type=str,
                            default="outputs/mmmu_pro/inference_results.jsonl",
                            help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str,
                            default="outputs/mmmu_pro/eval_results.csv",
                            help="Output file")

    args = parser.parse_args()

    if args.command == "infer":
        run_inference(args)
    elif args.command == "eval":
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
