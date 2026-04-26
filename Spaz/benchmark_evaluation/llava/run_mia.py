#!/usr/bin/env python3
"""
LLaVA-1.5-7B inference on MIA-Bench (Multimodal Instruction-following Ability Benchmark)
Source: https://github.com/apple/ml-mia-bench

MIA-Bench tests MLLMs' ability to strictly adhere to complex, compositional instructions.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import requests
from io import BytesIO


class LLaVAModel:
    """LLaVA model wrapper for MIA-Bench"""

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

    def generate(self, text, images, max_new_tokens=1024, temperature=0.01):
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


def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None


def run_inference(args):
    """Run inference on MIA-Bench"""
    print("="*60)
    print("LLaVA-1.5-7B MIA-Bench Inference")
    print("="*60)

    # Load MIA benchmark data
    print(f"Loading MIA benchmark from {args.data_file}...")
    with open(args.data_file, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Filter by type if specified
    if args.filter_type:
        data = [d for d in data if d['type'] == args.filter_type]
        print(f"Filtered to {args.filter_type} type: {len(data)} samples")

    # Limit samples if requested
    if args.max_samples and args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load model
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = LLaVAModel(args.model_path, device=device)

    # Run inference
    results = []
    failed_samples = []

    for i in tqdm(range(len(data)), desc="Running inference"):
        sample = data[i]
        image_url = sample['image']
        instruction = sample['instruction']

        try:
            # Load image
            image = load_image_from_url(image_url)
            if image is None:
                failed_samples.append((i, image_url, "Failed to load image"))
                result = {
                    "url": image_url,
                    "instruction": instruction,
                    "type": sample['type'],
                    "error": "Failed to load image"
                }
                results.append(result)
                continue

            # Generate response
            response = model.generate(
                instruction,
                [image],
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )

            if i < 3 or (i + 1) % 50 == 0:  # Print first 3 and every 50th
                print(f"\nSample {i}:")
                print(f"Instruction: {instruction[:100]}...")
                print(f"Response: {response[:150]}...")
                print('-' * 50)

            # Save result in MIA-Bench format
            result = {
                "url": image_url,
                "text": response,
                "instruction": instruction,  # Keep for reference
                "type": sample['type'],
                "components": sample['components'],
                "component_type": sample['component_type']
            }
            results.append(result)

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error processing sample {i} (URL={image_url}): {error_msg}")
            failed_samples.append((i, image_url, error_msg))

            result = {
                "url": image_url,
                "instruction": instruction,
                "type": sample['type'],
                "error": error_msg
            }
            results.append(result)
            continue

        # Save intermediate results every 20 samples
        if (i + 1) % 20 == 0:
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
        for i, url, err in failed_samples[:5]:
            print(f"  - Sample {i}: {err[:100]}")

    # Print statistics by type
    type_counts = {}
    for r in results:
        if 'error' not in r:
            t = r.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nSamples by type:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")


def main():
    parser = argparse.ArgumentParser(description="LLaVA-1.5-7B MIA-Bench Evaluation")
    parser.add_argument("--model-path", type=str,
                       default="/projects/bdpn/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
                       help="Path to LLaVA model")
    parser.add_argument("--data-file", type=str,
                       default="mia_benchmark_all.json",
                       help="MIA benchmark JSON file")
    parser.add_argument("--output-file", type=str,
                       default="outputs/mia/inference_results.jsonl",
                       help="Output file")
    parser.add_argument("--max-samples", type=int, help="Max samples to process")
    parser.add_argument("--filter-type", type=str,
                       choices=['basic', 'intermediate', 'advanced', 'creative', 'complex'],
                       help="Filter to specific instruction type")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Max tokens to generate (MIA needs longer responses)")
    parser.add_argument("--temperature", type=float, default=0.01,
                       help="Sampling temperature")
    parser.add_argument("--cpu", action="store_true",
                       help="Use CPU instead of GPU")

    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
