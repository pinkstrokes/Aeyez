#!/usr/bin/env python3
"""
MMMU PRO Evaluation with IterativeRefinementFlow

This script evaluates the IterativeRefinementFlow on the MMMU PRO benchmark dataset
using the unified BenchmarkInfer system for maximum code reuse and consistency.

MMMU PRO includes three subsets:
- "vision": Questions embedded in images (577 validation samples)
- "standard (4 options)": Traditional 4-choice questions (577 validation samples)
- "standard (10 options)": 10-choice questions (577 validation samples)
"""

import os
import sys
import argparse
import json
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

# Add the multi-agent directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

# Import MMMU PRO components
from dataset_utils import MMMUProDataset, format_options

# Import the unified base class
from app.utils.benchmark_infer import BenchmarkInfer

# Import result converter utilities (if available)
utils_path = current_dir.parent / "utils"
sys.path.insert(0, str(utils_path))
try:
    from result_converter import (
        generate_summary_csv_from_questions,
        generate_evaluation_jsonl_from_questions
    )
except ImportError:
    # Fallback implementations
    def generate_summary_csv_from_questions(session_dir, data, csv_output_file, dataset_name):
        print(f"⚠️ Summary CSV generation not available - skipping {csv_output_file}")

    def generate_evaluation_jsonl_from_questions(session_dir, data, jsonl_output_file, dataset_name):
        print(f"⚠️ Evaluation JSONL generation not available - skipping {jsonl_output_file}")


class MMMUProBenchmarkInfer(BenchmarkInfer):
    """
    MMMU PRO specific implementation of the BenchmarkInfer base class.

    Supports all three MMMU PRO subsets:
    - vision: Questions embedded in images
    - standard (4 options): Traditional 4-choice questions
    - standard (10 options): 10-choice questions with increased difficulty
    """

    def __init__(self):
        """Initialize MMMU PRO benchmark inference system."""
        super().__init__(
            benchmark_name="mmmu_pro",
            image_cache_dir=None  # Use default cache location
        )

        # MMMU PRO dataset loader
        self.dataset_loader = None

    def set_dataset_loader(self, loader: MMMUProDataset):
        """Set the dataset loader instance."""
        self.dataset_loader = loader

    # ==================== Abstract Method Implementations ====================

    def build_flow_input(self, sample: Dict[str, Any]) -> str:
        """
        Convert MMMU PRO sample to flow input format.

        Handles all three subsets with their specific formatting requirements.
        """
        subset = sample.get('subset', '')
        question = sample.get('question', '')
        options = sample.get('options', [])
        images = sample.get('images', [])

        # Format options if available
        options_text = ""
        if options:
            options_text = "\n\nOptions:\n" + format_options(options)

        if subset == "vision":
            # Vision subset: Question is embedded in image, but we may have text options
            if options_text:
                complete_question = f"Please answer the question shown in the image.{options_text}"
            else:
                complete_question = "Please answer the question shown in the image."
        else:
            # Standard subsets: Separate text question and options
            if options_text:
                complete_question = f"{question}{options_text}"
            else:
                complete_question = question

        # Handle images
        if images:
            # Save first image (MMMU PRO typically has one primary image)
            try:
                image_path = self._save_image_to_cache(images[0], sample.get('id', ''))
                flow_input = f"{complete_question}\nimage_path:{image_path}"
            except Exception as e:
                print(f"⚠️ Error saving image for sample {sample.get('id', '')}: {e}")
                flow_input = complete_question
        else:
            flow_input = complete_question

        return flow_input

    def _save_image_to_cache(self, image, sample_id: str) -> str:
        """
        Save PIL image to cache directory.

        Args:
            image: PIL Image object
            sample_id: Unique sample identifier

        Returns:
            Path to cached image file
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for image processing")
        # Generate cache filename from sample ID
        cache_name = hashlib.md5(f"mmmu_pro_{sample_id}".encode()).hexdigest() + ".jpg"
        cache_path = self.image_cache_dir / cache_name

        # Check if already cached
        if cache_path.exists():
            return str(cache_path)

        try:
            # Convert to RGB and save to cache
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large to avoid API limits
            max_size = (2048, 2048)  # Reasonable limit for most APIs
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

            image.save(cache_path, format='JPEG', quality=85)
            return str(cache_path)

        except Exception as e:
            print(f"❌ Error processing image for {sample_id}: {e}")
            raise

    def extract_question(self, sample: Dict[str, Any]) -> str:
        """Extract question text from MMMU PRO sample."""
        subset = sample.get('subset', '')
        if subset == "vision":
            return "[Question embedded in image]"
        return sample.get('question', '')

    def extract_options(self, sample: Dict[str, Any]) -> List[str]:
        """Extract answer options from MMMU PRO sample."""
        options = sample.get('options', [])
        if options:
            # Format with letter labels
            formatted = []
            for i, option in enumerate(options):
                letter = chr(ord('A') + i)
                formatted.append(f"({letter}) {option}")
            return formatted
        return []

    def extract_expected_answer(self, sample: Dict[str, Any]) -> str:
        """Extract expected answer from MMMU PRO sample."""
        return sample.get('answer', '')

    def extract_question_id(self, sample: Dict[str, Any]) -> str:
        """Extract unique question ID from MMMU PRO sample using validation split index."""
        # Use validation_index if available (set by the main loop), otherwise fall back to ID
        if 'validation_index' in sample:
            return f"validation_{sample['validation_index']}"
        else:
            # Fallback to original ID if validation_index is not available
            return sample.get('id', '')

    def extract_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from MMMU PRO sample for logging."""
        return {
            'mmmu_pro_id': sample.get('id', ''),
            'validation_index': sample.get('validation_index', ''),  # Validation split index
            'subset': sample.get('subset', ''),
            'subject': sample.get('subject', ''),
            'topic_difficulty': sample.get('topic_difficulty', ''),
            'num_options': len(sample.get('options', [])),
            'num_images': len(sample.get('images', [])),
            'dataset': 'mmmu_pro'
        }

    def load_dataset(self, subset_name: str, **kwargs) -> pd.DataFrame:
        """
        Load MMMU PRO dataset subset.

        Args:
            subset_name: Name of the MMMU PRO subset
            **kwargs: Additional arguments (max_samples, validation_only)

        Returns:
            pandas DataFrame with MMMU PRO samples
        """
        if not self.dataset_loader:
            raise ValueError("Dataset loader not initialized. Call set_dataset_loader() first.")

        # Get samples from the subset
        max_samples = kwargs.get('max_samples')
        validation_only = kwargs.get('validation_only', True)  # Default to validation only

        samples = self.dataset_loader.get_samples(
            subset_name,
            max_samples=max_samples,
            validation_only=validation_only
        )

        # Process samples to standardized format
        processed_samples = []
        for sample in samples:
            processed = self.dataset_loader.process_sample(sample, subset_name)
            processed_samples.append(processed)

        # Convert to DataFrame
        data = pd.DataFrame(processed_samples)
        print(f"📊 Loaded MMMU PRO {subset_name}: {len(data)} samples")
        return data

    def print_sample_info(self, sample: Dict[str, Any]):
        """Print MMMU PRO specific sample information."""
        print(f"ID: {sample.get('id', 'N/A')}")
        print(f"Subset: {sample.get('subset', 'N/A')}")
        print(f"Subject: {sample.get('subject', 'N/A')}")
        print(f"Difficulty: {sample.get('topic_difficulty', 'N/A')}")
        print(f"Options: {len(sample.get('options', []))}")
        print(f"Expected: {sample.get('answer', 'N/A')}")


def create_output_filename(subset: str, args) -> str:
    """Create output filename based on subset and arguments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean subset name for filename
    subset_clean = subset.replace(" ", "_").replace("(", "").replace(")", "")

    filename = f"mmmu_pro_{subset_clean}_{timestamp}.jsonl"
    return filename


def run_flow_inference(args):
    """Main inference pipeline for MMMU PRO."""
    print(f"🚀 Starting MMMU PRO evaluation with IterativeRefinementFlow")
    print(f"📋 Subset: {args.subset}")

    # Setup vLLM if needed
    if not args.skip_vllm_setup:
        try:
            from app.utils.vllm_setup import check_and_setup_vllm
            check_and_setup_vllm(vision_port=args.vision_port, text_port=args.text_port)
        except ImportError:
            print("⚠️ vLLM setup not available - proceeding without setup")

    # Initialize dataset loader
    print("📥 Loading MMMU PRO dataset...")
    dataset_loader = MMMUProDataset()

    # Load the specific subset
    try:
        dataset_loader.load_subsets([args.subset])
    except Exception as e:
        print(f"❌ Failed to load MMMU PRO subset '{args.subset}': {e}")
        return

    # Initialize adapter (uses config automatically)
    adapter = MMMUProBenchmarkInfer()
    print(f"🔄 Max iterations: {adapter.flow_executor.underlying_flow.max_iterations}")
    adapter.set_dataset_loader(dataset_loader)

    # Load dataset
    data = adapter.load_dataset(
        args.subset,
        max_samples=args.max_samples,
        validation_only=args.validation_only
    )

    if len(data) == 0:
        print("❌ No data loaded. Check subset name and validation settings.")
        return

    # Filter dataset if needed
    data = adapter.filter_dataset(
        data,
        start_sample=args.start_sample,
        end_sample=args.end_sample,
        max_samples=args.max_samples
    )

    # Start logging session
    experiment_config = adapter.build_experiment_config(args)
    experiment_config.update({
        'subset': args.subset,
        'total_samples': len(data),
        'validation_only': args.validation_only
    })

    adapter.start_logging_session(experiment_config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process samples
    print(f"🔄 Processing {len(data)} samples...")

    for i in tqdm(range(len(data)), desc="Running inference"):
        sample = data.iloc[i].to_dict()

        # Add validation_index for consistent log naming (like regular MMMU)
        # This represents the sample's position in the original dataset (preserves filtering)
        original_index = data.index[i]  # Get the original DataFrame index
        sample['validation_index'] = original_index

        start_time = datetime.now()
        response = adapter.execute_on_sample(sample)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Print execution summary if verbose
        if args.verbose:
            adapter.print_execution_summary(i, len(data), response, execution_time, sample)

        # Print completion status with validation index (consistent with regular MMMU)
        validation_idx = sample.get('validation_index', 'unknown')
        expected_answer = sample.get('answer', 'N/A')
        print(f"✅ Question {validation_idx} (validation index) completed | Expected: {expected_answer} | Time: {execution_time:.2f}s")

    # Finish logging session
    adapter.finish_logging_session()

    # Generate output files
    if adapter.session_dir:
        # Create output filename
        output_filename = create_output_filename(args.subset, args)
        output_path = output_dir / output_filename

        # Generate evaluation JSONL for the evaluator
        try:
            generate_evaluation_jsonl_from_questions(
                adapter.session_dir,
                data,
                str(output_path),
                f"mmmu_pro_{args.subset}"
            )
            print(f"📄 Results saved to: {output_path}")

            # Save session info for external tools
            adapter.save_session_info(str(output_path))

        except Exception as e:
            print(f"⚠️ Error generating output files: {e}")

    print(f"✅ MMMU PRO inference completed!")
    print(f"📊 Processed {len(data)} samples from {args.subset}")

    if adapter.session_dir:
        print(f"📁 Session logs: {adapter.session_dir}")
        
        # Auto-run evaluation after inference
        print(f"🔄 Auto-running evaluation on session results...")
        try:
            from evaluate_flow_results import MMMUProFlowEvaluator
            evaluator = MMMUProFlowEvaluator(args.output_dir)
            results = evaluator.evaluate(adapter.session_dir)
            print(f"✅ Auto-evaluation completed!")
            print(f"📈 Overall accuracy: {results['metrics']['overall']['accuracy']:.4f} ({results['metrics']['overall']['accuracy']*100:.2f}%)")
        except Exception as e:
            print(f"⚠️ Auto-evaluation failed: {e}")
            print("💡 You can manually run evaluation later with:")
            print(f"   python run_mmmu_pro_flow.py eval --results-dir {adapter.session_dir}")


def run_evaluation(args):
    """Run evaluation on existing results."""
    print(f"📊 Running evaluation on MMMU PRO results...")

    try:
        # Import and run the new flow evaluator
        from evaluate_flow_results import MMMUProFlowEvaluator

        evaluator = MMMUProFlowEvaluator(args.output_dir)
        results = evaluator.evaluate(args.results_dir)

        print(f"✅ Evaluation completed!")
        print(f"📈 Overall accuracy: {results['metrics']['overall']['accuracy']:.4f}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        # Fallback to original evaluator
        try:
            from evaluate import MMMUProEvaluator
            evaluator = MMMUProEvaluator(args.output_dir)
            results = evaluator.evaluate(args.input_file)
            print(f"✅ Fallback evaluation completed!")
            print(f"📈 Overall accuracy: {results['metrics']['overall']['accuracy']:.4f}")
        except Exception as e2:
            print(f"❌ Fallback evaluation also failed: {e2}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="MMMU PRO Flow Evaluation")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("subset",
                             choices=["vision", "standard (4 options)", "standard (10 options)"],
                             help="MMMU PRO subset to evaluate. Available options: 'vision' (questions embedded in images), 'standard (4 options)' (4-choice questions), 'standard (10 options)' (10-choice questions)")
    infer_parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    infer_parser.add_argument("--start-sample", type=int, help="Starting sample index")
    infer_parser.add_argument("--end-sample", type=int, help="Ending sample index (inclusive)")
    infer_parser.add_argument("--validation-only", action="store_true", default=True, help="Only process validation samples (default: True)")
    infer_parser.add_argument("--all-samples", action="store_true", help="Process all samples (not just validation)")
    infer_parser.add_argument("--output-dir", default="./output", help="Output directory for results")
    infer_parser.add_argument("--vision-port", type=int, default=8008, help="vLLM vision model port")
    infer_parser.add_argument("--text-port", type=int, default=8009, help="vLLM text model port")
    infer_parser.add_argument("--skip-vllm-setup", action="store_true", help="Skip vLLM cluster setup")
    infer_parser.add_argument("--verbose", action="store_true", help="Print detailed execution info")

    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation on existing results")
    eval_parser.add_argument("--results-dir", help="Directory containing validation_*.json files from flow inference")
    eval_parser.add_argument("--input-file", help="Input JSONL file with inference results (fallback)")
    eval_parser.add_argument("--output-dir", default="./output", help="Output directory for evaluation results")

    args = parser.parse_args()

    # Handle --all-samples flag
    if hasattr(args, 'all_samples') and args.all_samples:
        args.validation_only = False

    if args.mode == "infer":
        run_flow_inference(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()