#!/usr/bin/env python3
"""
MIA-Bench Inference Runner

This script runs inference on MIA-Bench dataset using our existing LLM infrastructure.
It generates results in the official MIA evaluation format.

Usage:
    python run_mia_inference.py --model qwen2_5_vl_3b --output outputs/qwen3b_results.jsonl
    python run_mia_inference.py --model gpt4o_mini --max-samples 50
    python run_mia_inference.py --model qwen2_5_vl_32b --start-sample 100 --end-sample 200
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
# Temporarily disable tqdm due to compatibility issues
TQDM_AVAILABLE = False
print("⚠️  Using basic progress tracking (tqdm disabled for compatibility)")
import traceback

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

# Import our model wrapper
from model_utils import MIAModelWrapper, MIAFlowWrapper, SUPPORTED_MODELS, list_supported_models


def load_mia_dataset(data_path: str = "data/instruction_benchmark_all.json") -> List[Dict[str, Any]]:
    """
    Load MIA-Bench dataset.
    
    Args:
        data_path: Path to the dataset JSON file
        
    Returns:
        List of MIA samples
    """
    full_path = current_dir / data_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"MIA dataset not found at {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f" Loaded MIA dataset: {len(data)} samples")
    return data


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save inference results in JSONL format (official MIA format).
    
    Args:
        results: List of result dictionaries
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f" Results saved to: {output_path}")
    print(f" Total samples processed: {len(results)}")


async def process_sample_with_retry(wrapper: MIAModelWrapper, sample: Dict[str, Any], 
                                  max_retries: int = 3) -> Dict[str, Any]:
    """
    Process a single sample with retry logic.
    
    Args:
        wrapper: Model wrapper instance
        sample: MIA sample to process
        max_retries: Maximum number of retries
        
    Returns:
        Result dictionary in MIA format
    """
    for attempt in range(max_retries):
        try:
            response = await wrapper.generate_response(sample["instruction"], sample["image"])
            return wrapper.format_for_mia_evaluation(sample, response)
        
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for sample {sample['image']}: {e}")
            if attempt == max_retries - 1:
                # Final attempt failed, return error response
                return wrapper.format_for_mia_evaluation(sample, "error")
            
            # Wait before retry
            await asyncio.sleep(2 ** attempt)  # Exponential backoff


async def run_inference(args):
    """
    Run MIA-Bench inference with specified configuration.
    
    Args:
        args: Command line arguments
    """
    print(f" Starting MIA-Bench inference")
    print(f"   Model: {args.model}")
    print(f"   Output: {args.output}")
    
    # Load dataset
    dataset = load_mia_dataset(args.data_path)
    
    # Apply sample range filtering
    if args.start_sample is not None or args.end_sample is not None:
        start = args.start_sample or 0
        end = args.end_sample or len(dataset)
        dataset = dataset[start:end]
        print(f" Processing samples {start} to {end-1} ({len(dataset)} samples)")
    elif args.max_samples is not None:
        dataset = dataset[:args.max_samples]
        print(f" Processing first {len(dataset)} samples")
    
    # Initialize model wrapper
    try:
        if args.use_flow:
            wrapper = MIAFlowWrapper(args.model, args.reasoning_model)
            print(" Using flow-ready wrapper (single-model mode)")
        else:
            wrapper = MIAModelWrapper(args.model)
            print(" Using direct model wrapper")
    except Exception as e:
        print(f" Failed to initialize model: {e}")
        return
    
    # Process samples
    results = []
    failed_samples = []
    
    print(f" Processing {len(dataset)} samples...")
    
    # Use semaphore to limit concurrent requests (especially for API models)
    semaphore = asyncio.Semaphore(args.concurrent_requests)
    
    async def process_with_semaphore(sample, idx):
        async with semaphore:
            try:
                if args.use_flow:
                    result = await wrapper.process_sample(sample)
                else:
                    result = await process_sample_with_retry(wrapper, sample, args.max_retries)
                
                # Add metadata for debugging
                result["sample_idx"] = idx
                result["model_config"] = args.model
                
                return result, None
            except Exception as e:
                error_info = {
                    "sample_idx": idx,
                    "image_url": sample.get("image", "unknown"),
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                return None, error_info
    
    # Process all samples concurrently with progress bar
    tasks = [process_with_semaphore(sample, idx) for idx, sample in enumerate(dataset)]
    
    if TQDM_AVAILABLE:
        # Use tqdm for progress tracking
        async for task in tqdm.as_completed(tasks, desc="Processing samples"):
            result, error = await task
            if result:
                results.append(result)
            else:
                failed_samples.append(error)
    else:
        # Fallback: process without progress bar
        print(f"Processing {len(tasks)} samples...")
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        for i, completed in enumerate(completed_tasks):
            if isinstance(completed, Exception):
                error_info = {
                    "sample_idx": i,
                    "image_url": dataset[i].get("image", "unknown"),
                    "error": str(completed),
                    "traceback": str(completed)
                }
                failed_samples.append(error_info)
            else:
                result, error = completed
                if result:
                    results.append(result)
                else:
                    failed_samples.append(error)
            
            # Simple progress indicator
            if (i + 1) % max(1, len(tasks) // 10) == 0:
                print(f"Processed {i + 1}/{len(tasks)} samples...")
    
    # Report results
    print(f"\n📊 Inference completed:")
    print(f"   ✅ Successful: {len(results)}")
    print(f"   ❌ Failed: {len(failed_samples)}")
    
    if failed_samples:
        print(f"\n  Failed samples:")
        for error in failed_samples[:5]:  # Show first 5 errors
            print(f"   Sample {error['sample_idx']}: {error['error']}")
        if len(failed_samples) > 5:
            print(f"   ... and {len(failed_samples) - 5} more")
    
    # Save results
    if results:
        save_results(results, args.output)
        
        # Save metadata
        metadata = {
            "model_config": args.model,
            "total_samples": len(dataset),
            "successful_samples": len(results),
            "failed_samples": len(failed_samples),
            "timestamp": datetime.now().isoformat(),
            "args": vars(args)
        }
        
        metadata_path = args.output.replace('.jsonl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f" Metadata saved to: {metadata_path}")
    
    else:
        print("❌ No successful results to save")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run MIA-Bench inference using our LLM infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Qwen2.5-VL-3B on first 50 samples
  python run_mia_inference.py --model qwen2_5_vl_3b --max-samples 50
  
  # Run with GPT-4o-mini on specific range
  python run_mia_inference.py --model gpt4o_mini --start-sample 100 --end-sample 200
  
  # Run full evaluation with Qwen2.5-VL-32B
  python run_mia_inference.py --model qwen2_5_vl_32b --output outputs/qwen32b_full.jsonl
  
  # List supported models
  python run_mia_inference.py --list-models
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default="mia_gpt4o_mini",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model configuration to use"
    )
    
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default="text_only_reasoning",
        help="Reasoning model config for flow mode (future use)"
    )
    
    parser.add_argument(
        "--use-flow",
        action="store_true",
        help="Use flow-ready wrapper (preparation for multi-agent)"
    )
    
    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/instruction_benchmark_all.json",
        help="Path to MIA dataset JSON file"
    )
    
    # Sample range configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process"
    )
    
    parser.add_argument(
        "--start-sample",
        type=int,
        help="Starting sample index"
    )
    
    parser.add_argument(
        "--end-sample", 
        type=int,
        help="Ending sample index (exclusive)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file path (auto-generated if not specified)"
    )
    
    # Performance configuration
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=5,
        help="Number of concurrent requests (reduce for API rate limits)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per sample"
    )
    
    # Utility options
    parser.add_argument(
        "--list-models",
        action="store_true", 
        help="List supported model configurations"
    )
    
    args = parser.parse_args()
    
    # Handle utility options
    if args.list_models:
        list_supported_models()
        return
    
    # Auto-generate output path if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_suffix = ""
        if args.max_samples:
            sample_suffix = f"_max{args.max_samples}"
        elif args.start_sample is not None or args.end_sample is not None:
            start = args.start_sample or 0
            end = args.end_sample or "end"
            sample_suffix = f"_range{start}to{end}"
        
        args.output = f"outputs/{args.model}_mia_results_{timestamp}{sample_suffix}.jsonl"
    
    # Validate sample range
    if args.start_sample is not None and args.end_sample is not None:
        if args.start_sample >= args.end_sample:
            print("❌ Error: start-sample must be less than end-sample")
            return
    
    if args.max_samples is not None and (args.start_sample is not None or args.end_sample is not None):
        print("❌ Error: Cannot use --max-samples with --start-sample/--end-sample")
        return
    
    # Run inference (Python 3.6 compatibility)
    try:
        # Python 3.7+ has asyncio.run, 3.6 needs manual event loop
        try:
            asyncio.run(run_inference(args))
        except AttributeError:
            # Fallback for Python 3.6
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_inference(args))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
