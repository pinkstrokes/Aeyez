#!/usr/bin/env python3
"""
MIA-Bench Evaluation with IterativeRefinementFlow (v2)

This script integrates the MIA-Bench evaluation pipeline with the multi-agent flow system
using the unified BenchmarkInfer base class for maximum code reuse.
"""

import json
import argparse
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add the multi-agent directory to Python path for imports
import sys
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

# Import the unified base class
from app.utils.benchmark_infer import BenchmarkInfer


class MIABenchmarkInfer(BenchmarkInfer):
    """
    MIA-Bench specific implementation of the BenchmarkInfer base class.

    Only requires implementing the abstract methods - all common functionality
    is inherited from the parent class.
    """

    def __init__(self):
        """Initialize MIA benchmark inference system."""
        super().__init__(
            benchmark_name="mia",
            image_cache_dir=None  # Use default cache location
        )

    # ==================== Abstract Method Implementations ====================

    def build_flow_input(self, sample: Dict[str, Any]) -> str:
        """
        Convert MIA sample to flow input format.

        Args:
            sample: MIA sample dict with 'instruction' and 'image'

        Returns:
            Formatted flow input string
        """
        instruction = sample['instruction']
        image_url = sample['image']

        # Process image and get local path
        try:
            image_path = self.load_and_cache_image(image_url)
            # Create flow input format with image
            flow_input = f"{instruction}\nimage_path:{image_path}"
        except Exception as e:
            print(f"⚠️ Failed to process image {image_url}: {e}")
            # Fallback to text-only processing
            flow_input = instruction

        return flow_input

    def extract_question(self, sample: Dict[str, Any]) -> str:
        """Extract question text from MIA sample."""
        return sample.get('instruction', '')

    def extract_options(self, sample: Dict[str, Any]) -> List[str]:
        """Extract answer options from MIA sample (MIA doesn't have multiple choice)."""
        return []

    def extract_expected_answer(self, sample: Dict[str, Any]) -> str:
        """Extract expected answer from MIA sample (MIA doesn't have ground truth)."""
        return ""

    def extract_question_id(self, sample: Dict[str, Any]) -> str:
        """Extract unique question ID from MIA sample (use current index)."""
        # Use the current_index if available, otherwise fall back to image URL
        if 'current_index' in sample:
            return f"question_{sample['current_index']}"
        else:
            # Fallback to image URL for backward compatibility
            image_url = sample.get('image', '')
            import re
            sanitized_id = re.sub(r'[<>:"/\\|?*]', '_', image_url)
            return sanitized_id

    def extract_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from MIA sample for logging."""
        return {
            'mia_image_url': sample.get('image', ''),
            'instruction_length': len(sample.get('instruction', '')),
            'benchmark': 'mia'
        }

    def load_dataset(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        Load MIA-Bench dataset and convert to DataFrame.

        Args:
            data_path: Path to the dataset JSON file

        Returns:
            pandas DataFrame with MIA samples
        """
        full_path = current_dir / data_path

        if not full_path.exists():
            raise FileNotFoundError(f"MIA dataset not found at {full_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to DataFrame for consistency
        df = pd.DataFrame(data)

        # Add index as ID if not present
        if 'id' not in df.columns:
            df['id'] = df['image']  # Use image URL as ID

        print(f"📊 Loaded MIA dataset: {len(df)} samples")
        return df

    def print_sample_info(self, sample: Dict[str, Any]):
        """Print MIA-specific sample information."""
        print(f"🖼️ Image: {sample.get('image', 'N/A')}")
        instruction = sample.get('instruction', '')
        if len(instruction) > 100:
            print(f"📝 Instruction: {instruction[:100]}...")
        else:
            print(f"📝 Instruction: {instruction}")


def run_flow_inference(args):
    """
    Run inference using IterativeRefinementFlow on MIA dataset.
    Uses the unified BenchmarkInfer system for maximum code reuse.
    """
    print("🚀 Starting MIA evaluation with IterativeRefinementFlow")
    print("="*70)

    # Check and set up vLLM if needed
    # Get port configuration
    vision_port = getattr(args, 'vision_port', 8006)
    text_port = getattr(args, 'text_port', 8007)

    # Update config to use the specified ports
    try:
        from app.config import config
        if hasattr(config, 'llm') and 'translator_api' in config.llm:
            config.llm['translator_api'].base_url = f"http://localhost:{vision_port}/v1"
        if hasattr(config, 'llm') and 'reasoning_api' in config.llm:
            config.llm['reasoning_api'].base_url = f"http://localhost:{text_port}/v1"
        print(f"🔧 Updated model configs to use ports {vision_port} (vision) and {text_port} (text)")
    except Exception as e:
        print(f"⚠️ Could not update model configs: {e}")

    if not getattr(args, 'skip_vllm_setup', False):
        try:
            from app.utils.vllm_setup import check_and_setup_vllm
            check_and_setup_vllm(vision_port=vision_port, text_port=text_port)
        except ImportError:
            print("⚠️ vLLM setup not available - proceeding without cluster setup")
    else:
        print("⏭️ Skipping vLLM setup as requested")
    print()

    # Initialize MIA benchmark inference system (uses config automatically)
    mia_infer = MIABenchmarkInfer()
    print(f"🔄 Max iterations: {mia_infer.flow_executor.underlying_flow.max_iterations}")

    # Load dataset
    print("📊 Loading MIA dataset...")
    data = mia_infer.load_dataset(args.data_path)

    # Filter dataset based on arguments
    data = mia_infer.filter_dataset(
        data,
        start_sample=getattr(args, 'start_sample', None),
        end_sample=getattr(args, 'end_sample', None),
        max_samples=getattr(args, 'max_samples', None)
    )

    # Build experiment configuration and start logging session
    experiment_config = mia_infer.build_experiment_config(args)
    experiment_config.update({
        'data_path': args.data_path,
        'total_samples': len(data)
    })

    mia_infer.start_logging_session(experiment_config=experiment_config)

    print(f"📁 Session directory: {mia_infer.session_dir}")
    print(f"📄 Individual question results will be saved as JSON files")

    # Process samples using the unified execution system
    for i in tqdm(range(len(data)), desc="Running flow inference"):
        sample_row = data.iloc[i]
        sample = mia_infer.clean_sample_data(sample_row.to_dict())

        # Add current index to sample for sequential question naming
        sample['current_index'] = i

        # Record start time
        start_time = datetime.now()

        # Execute flow using unified system
        response = mia_infer.execute_on_sample(sample)

        # Record end time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Print execution summary using unified system
        mia_infer.print_execution_summary(i, len(data), response, execution_time, sample)

        print(f"✅ Question {i+1}/{len(data)} completed | Time: {execution_time:.2f}s")

    print(f"✅ Flow inference completed. All results saved in session directory: {mia_infer.session_dir}")

    # Save session directory before finishing session
    session_directory = mia_infer.session_dir

    # Finish logging session
    mia_infer.finish_logging_session()

    # Generate summary files if utilities are available
    print("\n📄 Generating summary files from question logs...")
    try:
        # Try to import result conversion utilities
        utils_path = current_dir.parent / "utils"
        sys.path.insert(0, str(utils_path))

        from result_converter import (
            generate_summary_csv_from_questions,
            generate_evaluation_jsonl_from_questions
        )

        if session_directory:
            csv_output_file = session_directory / "inference_summary.csv"
            jsonl_output_file = session_directory / "inference_results.jsonl"

            generate_summary_csv_from_questions(session_directory, data, csv_output_file, "mia")
            generate_evaluation_jsonl_from_questions(session_directory, data, jsonl_output_file, "mia")

            print(f"✅ Summary CSV saved to {csv_output_file}")
            print(f"✅ Evaluation JSONL saved to {jsonl_output_file}")
        else:
            print("⚠️ No session directory available for summary generation")

    except ImportError:
        print("⚠️ Result conversion utilities not available - raw results saved in session directory")
        csv_output_file = "See session directory for individual question files"
        jsonl_output_file = "See session directory for individual question files"

    print(f"\n✅ Inference completed. Use MIA evaluation tools for assessment.")
    print(f"📁 All files saved in session directory: {mia_infer.session_dir}")

    # Save session info for external reference
    mia_infer.save_session_info()

    return mia_infer.session_dir


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MIA-Bench Evaluation with IterativeRefinementFlow (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with sample limits
  python run_mia_inference_flow_v2.py --max-samples 10

  # Custom port configuration
  python run_mia_inference_flow_v2.py --max-samples 5 --vision-port 8002 --text-port 8003

  # Process specific sample range
  python run_mia_inference_flow_v2.py --start-sample 100 --end-sample 200

  # Full evaluation with custom iterations
  python run_mia_inference_flow_v2.py --max-iterations 2
        """
    )

    # Dataset configuration
    parser.add_argument("--data-path", type=str, default="data/instruction_benchmark_all.json",
                       help="Path to MIA dataset JSON file")

    # Sample range configuration
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--start-sample", type=int, help="Starting sample index (0-based)")
    parser.add_argument("--end-sample", type=int, help="Ending sample index (inclusive)")

    # System configuration
    parser.add_argument("--skip-vllm-setup", action="store_true",
                       help="Skip automatic vLLM cluster setup (assume already configured)")
    parser.add_argument("--vision-port", type=int, default=8000,
                       help="Port for vision model API (default: 8000)")
    parser.add_argument("--text-port", type=int, default=8001,
                       help="Port for text model API (default: 8001)")

    # Logging configuration
    parser.add_argument("--benchmark-name", type=str, default="mia",
                       help="Benchmark name for logging organization (default: mia)")

    args = parser.parse_args()

    # Validate arguments
    if args.start_sample is not None and args.end_sample is not None:
        if args.start_sample >= args.end_sample:
            print("❌ Error: start-sample must be less than end-sample")
            return

    if args.max_samples is not None and (args.start_sample is not None or args.end_sample is not None):
        print("❌ Error: Cannot use --max-samples with --start-sample/--end-sample")
        return

    # Run inference
    session_dir = run_flow_inference(args)
    if session_dir:
        print(f"\n🎯 Session completed successfully!")
        print(f"📁 Session directory: {session_dir}")


if __name__ == "__main__":
    main()