#!/usr/bin/env python3
"""
MMMU Evaluation with IterativeRefinementFlow (v2)

This script evaluates the IterativeRefinementFlow on the MMMU benchmark dataset
using the unified BenchmarkInfer system for maximum code reuse and consistency.
"""

import os
import sys
import asyncio
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Add the multi-agent directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

# Import existing MMMU components - reuse everything possible
from dataset_utils import load_dataset, dump_image, MMMU_preproc

# Import the unified base class
from app.utils.benchmark_infer import BenchmarkInfer

# Import router and Level-1 runner (optional — only used when --use-router is set)
try:
    from app.utils.router import route_sample
    from app.utils.level1_runner import run_level1
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    print("⚠️  Router modules not available — --use-router will be disabled")

# Import result converter utilities
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

def load_validation_mapping(mapping_file: str = "validation_index_mapping.json") -> dict:
    """
    Load the mapping from full dataset index to validation split index.

    Returns:
        dict: Mapping from full_dataset_index -> validation_split_index
    """
    # Try project root first
    project_root = Path(__file__).parent.parent.parent.parent
    mapping_path = project_root / mapping_file

    # Also try current working directory as fallback
    if not mapping_path.exists():
        mapping_path = Path(mapping_file)

    if not mapping_path.exists():
        print(f"⚠️  Validation mapping file not found at: {mapping_path}")
        print(f"   Run: python create_validation_mapping.py from project root")
        return {}

    try:
        with open(mapping_path, 'r') as f:
            full_mapping = json.load(f)

        # Convert to simple int->int mapping
        validation_mapping = {}
        for full_idx_str, info in full_mapping.items():
            full_idx = int(full_idx_str)
            validation_idx = info['validation_index']
            validation_mapping[full_idx] = validation_idx

        print(f"✅ Loaded validation mapping: {len(validation_mapping)} entries")
        return validation_mapping

    except Exception as e:
        print(f"❌ Error loading validation mapping: {e}")
        return {}

class MMMUBenchmarkInfer(BenchmarkInfer):
    """
    MMMU-Bench specific implementation of the BenchmarkInfer base class.

    This refactored version provides the same functionality as the original
    run_mmmu_flow.py but with 73% less code by using the unified infrastructure.
    """

    def __init__(self):
        """Initialize MMMU benchmark inference system."""
        super().__init__(
            benchmark_name="mmmu",
            image_cache_dir=None  # Use default cache location
        )

        # MMMU-specific image dumping function
        self.dump_image_func = None

    def set_dump_image(self, func):
        """Set the image dumping function (reuse existing MMMU interface)."""
        self.dump_image_func = func

    # ==================== Abstract Method Implementations ====================

    def build_flow_input(self, sample: dict) -> str:
        """
        Convert MMMU sample to flow input format.
        Reuses existing data processing logic from the original implementation.

        Options are formatted as multi-line "A. text\nB. text" so that
        iterative_refinement._parse_options() takes the multi-line path and
        strips the letter prefix cleanly — preventing double-labeling like
        "A. (A) text" when _format_options() re-adds the letter.
        """
        question = sample['question']
        choices = []

        # Build choices list as "A. text" (no extra "(A)" wrapper)
        for ch in ['A', 'B', 'C', 'D']:
            if ch in sample and not pd.isna(sample[ch]):
                choices.append(f"{ch}. {sample[ch]}")

        # Format complete question with options on separate lines
        # so _parse_options takes the '\n in options_text' path
        if choices:
            options_block = "\n".join(choices)
            complete_question = f"{question}\n\nOptions:\n{options_block}"
        else:
            complete_question = question

        # Handle images if present
        image_path = None
        if 'image' in sample:
            image_val = sample['image']
            has_image = False

            if isinstance(image_val, (list, np.ndarray)):
                has_image = len(image_val) > 0 and not all(pd.isna(img) for img in image_val)
            else:
                has_image = not pd.isna(image_val)

            if has_image and self.dump_image_func is not None:
                image_paths = self.dump_image_func(sample)
                if image_paths:
                    image_path = image_paths[0]

        # Create flow input format
        if image_path:
            flow_input = f"{complete_question}\nimage_path:{image_path}"
        else:
            flow_input = complete_question

        return flow_input

    def extract_question(self, sample: dict) -> str:
        """Extract question text from MMMU sample."""
        return sample.get('question', '')

    def extract_options(self, sample: dict) -> list:
        """Extract answer options from MMMU sample."""
        options = []
        for ch in ['A', 'B', 'C', 'D']:
            if ch in sample and not pd.isna(sample[ch]):
                options.append(f"({ch}) {sample[ch]}")
        return options

    def extract_expected_answer(self, sample: dict) -> str:
        """Extract expected answer from MMMU sample."""
        return sample.get('answer', '')

    def extract_question_id(self, sample: dict) -> str:
        """Extract unique question ID from MMMU sample using validation split index."""
        # Use validation_index if available (set by the main loop), otherwise fall back to ID
        if 'validation_index' in sample:
            # Determine prefix based on sample type
            if sample.get('id', '').startswith('dev_'):
                return f"dev_{sample['validation_index']}"
            else:
                return f"validation_{sample['validation_index']}"
        else:
            # Fallback to original ID if validation_index is not available
            return sample.get('id', '')

    def extract_metadata(self, sample: dict) -> dict:
        """Extract metadata from MMMU sample for logging."""
        return {
            'mmmu_id': sample.get('id', ''),
            'validation_index': sample.get('validation_index', ''),  # Validation split index
            'full_dataset_index': sample.get('index', ''),  # Full dataset index for reference
            'category': sample.get('category', ''),
            'subject': sample.get('subject', ''),
            'dataset': 'mmmu'
        }

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load MMMU dataset using existing utilities.

        Args:
            dataset_name: Name of the MMMU dataset

        Returns:
            pandas DataFrame with MMMU samples
        """
        # Reuse existing MMMU dataset loader
        data = load_dataset(dataset_name)
        print(f"📊 Loaded MMMU dataset: {len(data)} samples")
        return data

    def print_sample_info(self, sample: dict):
        """Print MMMU-specific sample information."""
        print(f"🆔 ID: {sample.get('id', 'N/A')}")
        print(f"📚 Category: {sample.get('category', 'N/A')}")
        print(f"🔬 Subject: {sample.get('subject', 'N/A')}")
        print(f"✅ Expected: {sample.get('answer', 'N/A')}")


def run_flow_inference(args):
    """
    Run inference using IterativeRefinementFlow on MMMU dataset.

    This function provides the same functionality as the original implementation
    but with dramatically reduced code complexity through the unified system.
    """
    print("🚀 Starting MMMU evaluation with IterativeRefinementFlow")
    print("="*70)

    # Check and set up vLLM if needed
    if not getattr(args, 'skip_vllm_setup', False):
        try:
            from app.utils.vllm_setup import check_and_setup_vllm
            vision_port = getattr(args, 'vision_port', 8004)
            text_port = getattr(args, 'text_port', 8005)
            check_and_setup_vllm(vision_port=vision_port, text_port=text_port)
        except ImportError:
            print("⚠️ vLLM setup not available - proceeding without cluster setup")
    else:
        print("⏭️ Skipping vLLM setup as requested")
    print()

    # Initialize MMMU benchmark inference system (uses config automatically)
    mmmu_infer = MMMUBenchmarkInfer()
    print(f"🔄 Max iterations: {mmmu_infer.flow_executor.underlying_flow.max_iterations}")

    # Set up image root directory and dump function (reuse existing logic)
    img_root = os.path.join(os.environ['LMUData'], 'images', 'MMMU')
    os.makedirs(img_root, exist_ok=True)

    def dump_image_func(line):
        return dump_image(line, img_root)

    mmmu_infer.set_dump_image(dump_image_func)

    # Load dataset
    print("📊 Loading MMMU dataset...")
    data = mmmu_infer.load_dataset(args.dataset)

    # Apply filtering if requested
    validation_only = getattr(args, 'validation_only', False)
    dev_only = getattr(args, 'dev_only', False)
    
    if validation_only and dev_only:
        print("❌ Error: Cannot use both --validation-only and --dev-only at the same time")
        return None
    elif validation_only:
        original_size = len(data)
        # Filter to only validation samples (IDs starting with 'validation_')
        data = data[data['id'].str.startswith('validation_')]
        filtered_size = len(data)
        print(f"🔍 Filtered to validation-only samples: {filtered_size} samples (from {original_size} total)")
    elif dev_only:
        original_size = len(data)
        # Filter to only dev samples (split='dev')
        data = data[data['split'] == 'dev']
        filtered_size = len(data)
        print(f"🔍 Filtered to dev-only samples: {filtered_size} samples (from {original_size} total)")
    else:
        print(f"📊 Using full dataset: {len(data)} samples")

    # Convert open questions to 2-choice format (A=correct answer, B='Other Answers')
    # so Reasoning Agent can match answer letters during evaluation
    data = MMMU_preproc(data)

    # Store original start sample for question indexing
    original_start_sample = getattr(args, 'start_sample', 0) or 0

    # Filter dataset based on arguments
    data = mmmu_infer.filter_dataset(
        data,
        start_sample=getattr(args, 'start_sample', None),
        end_sample=getattr(args, 'end_sample', None),
        max_samples=getattr(args, 'max_samples', None)
    )

    # Build experiment configuration and start logging session
    experiment_config = mmmu_infer.build_experiment_config(args)
    experiment_config.update({
        'dataset': args.dataset,
        'total_samples': len(data),
        'output_file': args.output_file if hasattr(args, 'output_file') else None
    })

    mmmu_infer.start_logging_session(experiment_config=experiment_config)

    # Get session directory for output files
    session_dir = mmmu_infer.session_dir

    print(f"📁 Session directory: {session_dir}")
    print(f"📄 Individual question results will be saved as JSON files")

    # Load validation mapping for proper indexing
    print("📋 Loading validation index mapping...")
    validation_mapping = load_validation_mapping()
    if not validation_mapping:
        print("⚠️  Warning: Could not load validation mapping, using fallback naming")

    # Determine router mode
    use_router = getattr(args, 'use_router', False) and ROUTER_AVAILABLE
    if getattr(args, 'use_router', False) and not ROUTER_AVAILABLE:
        print("⚠️  --use-router requested but router modules not available — falling back to Level 2 for all")
    if use_router:
        print("🔀 Router mode ENABLED — questions will be routed to Level 1 (1-shot) or Level 2 (iterative)")
    else:
        print("🔀 Router mode DISABLED — all questions run full Level 2 iterative loop")

    # Process samples using the unified execution system
    for i in tqdm(range(len(data)), desc="Running flow inference"):
        sample_row = data.iloc[i]
        sample = mmmu_infer.clean_sample_data(sample_row.to_dict())

        # Map full dataset index to validation split index
        full_dataset_index = sample.get('index', None)
        if full_dataset_index and validation_mapping and int(full_dataset_index) in validation_mapping:
            validation_index = validation_mapping[int(full_dataset_index)]
            sample['validation_index'] = validation_index
        else:
            # Fallback: use the old question_index method
            actual_question_index = original_start_sample + i
            sample['validation_index'] = actual_question_index
            if not validation_mapping:
                print(f"⚠️  Using fallback index {actual_question_index} for sample {i}")
            elif full_dataset_index:
                print(f"⚠️  Full dataset index {full_dataset_index} not found in validation mapping")

        # Record start time
        start_time = datetime.now()

        # ---- Extract base64 image (always, not just for router) ----
        # MMMU guarantees at least one image per question. We extract it
        # once here so both the router and Level 1 runner share the same ref.
        base64_img = None
        img_val = sample.get('image')
        if isinstance(img_val, list) and img_val:
            candidate = img_val[0]
            base64_img = candidate if isinstance(candidate, str) and len(candidate) > 100 else None
        elif isinstance(img_val, str) and len(img_val) > 100:
            base64_img = img_val

        if base64_img is None:
            print(f"⚠️  Warning: no valid base64 image found for sample {i} — forcing Level 2")

        # ---- Router branching ----
        router_result = None
        execution_level = 2  # default: full iterative loop

        if use_router:
            router_result = asyncio.run(route_sample(sample, base64_img))
            execution_level = router_result['level']
            conf = router_result.get('confidence', 0.0)
            reason = router_result.get('reason', '')
            print(f"🔀 Routed → Level {execution_level} (conf={conf:.2f}): {reason}")

            # Safety: Level 1 requires an image; fallback to Level 2 if missing
            if execution_level == 1 and base64_img is None:
                print("⚠️  Level 1 requested but no image available — falling back to Level 2")
                execution_level = 2
                if router_result:
                    router_result['level'] = 2
                    router_result['reason'] += ' [fallback: no image]'

        # Build per-question metadata (carries router info for logging)
        question_metadata = mmmu_infer.extract_metadata(sample)
        question_metadata['router'] = router_result if router_result else {'level': 2, 'note': 'router disabled'}
        question_metadata['execution_level'] = execution_level

        if execution_level == 1:
            # Level 1: 1-shot VLM caption + 1-shot text reasoning
            question = mmmu_infer.extract_question(sample)
            options = mmmu_infer.extract_options(sample)
            expected_answer = mmmu_infer.extract_expected_answer(sample)
            custom_question_id = mmmu_infer.extract_question_id(sample)

            l1_result = asyncio.run(run_level1(
                sample=sample,
                base64_image=base64_img,
            ))

            answer = l1_result.get('answer', '')
            caption = l1_result.get('caption', '')
            # model_response 只存干净的答案字母，与 Level 2 格式保持一致
            # 这样 eval 的规则提取器 can_infer_option() 能直接匹配，
            # 不会因 Visual Description 里出现 A/B/C/D 字母而误判
            response = answer  # e.g. "B"

            # Assemble token breakdown: router + vlm_step + text_step
            l1_token = l1_result.get('token_usage', {})
            router_token = router_result.get('token_usage', {}) if router_result else {}
            token_breakdown = {
                'router': router_token,
                'level1_vlm_step': l1_token.get('vlm_step', {}),
                'level1_text_step': l1_token.get('text_step', {}),
                'level1_total': l1_token.get('total', {}),
                'execution_total': l1_token.get('total', {}),
            }
            question_metadata['token_usage_breakdown'] = token_breakdown
            question_metadata['level1_success'] = l1_result.get('success', False)
            question_metadata['level1_caption'] = caption  # Visual Description 存进 metadata
            question_metadata['level1_raw_text_output'] = l1_result.get('raw_text_output', '')
            if l1_result.get('error'):
                question_metadata['level1_error'] = l1_result['error']

            # Log directly via log_save (bypasses FlowExecutor)
            if mmmu_infer.session_id:
                clean_meta = mmmu_infer._convert_numpy_types(question_metadata)
                mmmu_infer.log_save.start_individual_question(
                    question=question,
                    options=options,
                    expected_answer=expected_answer,
                    question_metadata=clean_meta,
                    custom_question_id=custom_question_id,
                )
                end_time_l1 = datetime.now()
                mmmu_infer.log_save.finish_individual_question(
                    model_response=response,
                    evaluation_results={'execution_time': (end_time_l1 - start_time).total_seconds()},
                    token_usage=token_breakdown,
                    critical_errors=[],
                )
        else:
            # Level 2: full iterative refinement loop (existing path)
            if router_result:
                # Attach router token info to metadata so it's logged alongside
                question_metadata['token_usage_breakdown'] = {
                    'router': router_result.get('token_usage', {}),
                }
            response = mmmu_infer.execute_on_sample(sample, question_metadata=question_metadata)

        # Record end time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Print execution summary using unified system
        mmmu_infer.print_execution_summary(i, len(data), response, execution_time, sample)

        validation_idx = sample.get('validation_index', 'unknown')
        print(f"✅ Question {validation_idx} (validation index) completed | Level {execution_level} | Expected: {sample['answer']} | Time: {execution_time:.2f}s")

    print(f"✅ Flow inference completed. All results saved in session directory: {session_dir}")

    # Finish logging session
    mmmu_infer.finish_logging_session()

    # Generate both summary CSV and evaluation-compatible JSONL
    print("\n📄 Generating summary files from question logs...")
    try:
        csv_output_file = session_dir / "inference_summary.csv"
        jsonl_output_file = session_dir / "inference_results.jsonl"

        generate_summary_csv_from_questions(session_dir, data, csv_output_file, args.dataset)
        generate_evaluation_jsonl_from_questions(session_dir, data, jsonl_output_file, args.dataset)

        print(f"✅ Summary CSV saved to {csv_output_file}")
        print(f"✅ Evaluation JSONL saved to {jsonl_output_file}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to generate summary files: {e}")
        csv_output_file = "See session directory for individual question files"
        jsonl_output_file = "See session directory for individual question files"

    print(f"\n✅ Inference completed. Use 'eval' mode with official MMMU evaluation for accuracy metrics.")
    print(f"📁 All files saved in session directory: {session_dir}")
    print(f"📊 Summary CSV: {csv_output_file}")
    print(f"📄 Evaluation JSONL: {jsonl_output_file}")

    # Save session directory path for SBATCH script reference
    session_info_file = Path("session_info.txt")
    with session_info_file.open('w') as f:
        f.write(f"SESSION_DIR={session_dir}\n")
        f.write(f"CSV_FILE={csv_output_file}\n")
        f.write(f"JSONL_FILE={jsonl_output_file}\n")

    return session_dir  # Return session directory for potential script usage


def main():
    """Main function with argument parsing (maintains same interface as original)."""
    parser = argparse.ArgumentParser(description="MMMU Evaluation with IterativeRefinementFlow")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Flow inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with IterativeRefinementFlow")
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    infer_parser.add_argument("--data-dir", type=str,
                             default="/u/hli36/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/inputs",
                             help="The absolute path of MMMU_DEV_VAL.tsv")
    infer_parser.add_argument("--validation-only", action="store_true",
                             help="Only load samples with IDs starting with 'validation_' (skip dev_ samples)")
    infer_parser.add_argument("--dev-only", action="store_true",
                             help="Only load samples with split='dev' (skip validation samples)")
    infer_parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    infer_parser.add_argument("--start-sample", type=int, help="Starting sample index (0-based)")
    infer_parser.add_argument("--end-sample", type=int, help="Ending sample index (inclusive)")
    infer_parser.add_argument("--skip-vllm-setup", action="store_true",
                             help="Skip automatic vLLM cluster setup (assume already configured)")
    infer_parser.add_argument("--benchmark-name", type=str, default="mmmu",
                             help="Benchmark name for logging organization (default: mmmu)")
    infer_parser.add_argument("--vision-port", type=int, default=8004,
                             help="Port for vision model API (default: 8004)")
    infer_parser.add_argument("--text-port", type=int, default=8005,
                             help="Port for text model API (default: 8005)")
    infer_parser.add_argument("--use-router", action="store_true",
                             help="Enable difficulty-based routing: easy questions → Level 1 (1-shot), "
                                  "hard questions → Level 2 (full iterative loop)")

    # Evaluation parser (reuse from run_mmmu.py) - import when needed
    eval_parser = subparsers.add_parser("eval", help="Run official MMMU evaluation on flow results")
    eval_parser.add_argument("--data-dir", type=str,
                            default="/u/hli36/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/inputs",
                            help="The absolute path of dataset")
    eval_parser.add_argument("--input-file", type=str,
                            default="/u/hli36/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/outputs/inference_results.jsonl",
                            help="Input file with flow inference results")
    eval_parser.add_argument("--output-file", type=str,
                            default="/u/hli36/MPU-RL/src/multi-agent/benchmark_evaluation/mmmu/outputs/flow_eval_results.csv",
                            help="Output file path")
    eval_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="qwen-flash",
                            choices=["gpt-3.5-turbo-0125","gpt-4-0125-preview","qwen-flash","qwen-plus"],
                            help="Model to use for evaluation")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"],
                            help="API type to use for evaluation")

    args = parser.parse_args()

    # Set environment variable (reuse existing pattern)
    os.environ['LMUData'] = args.data_dir

    if args.mode == "infer":
        session_dir = run_flow_inference(args)
        if session_dir:
            print(f"\n🎯 Session completed successfully!")
            print(f"📁 Session directory: {session_dir}")
    elif args.mode == "eval":
        # Import and run evaluation (reuse existing implementation)
        from run_mmmu import run_evaluation
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()