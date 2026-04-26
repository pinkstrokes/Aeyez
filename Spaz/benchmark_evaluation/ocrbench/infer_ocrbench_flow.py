#!/usr/bin/env python3
"""
OCRBench Evaluation with IterativeRefinementFlow

This script integrates OCRBench evaluation with the multi-agent flow system
using the unified BenchmarkInfer base class for maximum code reuse.

OCRBench supports various OCR tasks including:
- Text recognition, VQA, document parsing, table parsing, chart parsing
- Key information extraction, text spotting, formula recognition
- Multi-language support (English and Chinese)
"""

import os
import sys
import json
import base64
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import io
import re

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

# OCRBench evaluation modules
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Add the multi-agent directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

# Import the unified base class
from app.utils.benchmark_infer import BenchmarkInfer

# Import OCRBench evaluation functions
sys.path.append(os.path.join(current_dir, "OCRBench_v2_eval/eval_scripts"))
try:
    from vqa_metric import vqa_evaluation, cn_vqa_evaluation, math_expression_evaluation, vqa_evaluation_case_sensitive, counting_evaluation, cn_math_expression_evaluation
    from IoUscore_metric import vqa_with_position_evaluation, calculate_iou, extract_coordinates
    from TEDS_metric import TEDS, convert_markdown_table_to_html, convert_str_to_dict, convert_str_to_multi_dict, generate_combinations, dict_to_html, compute_f1_score, doc_parsing_evaluation, wrap_html_table
    from page_ocr_metric import cal_per_metrics
    from spotting_metric import extract_bounding_boxes_robust, spotting_evaluation
    OCRBENCH_EVAL_AVAILABLE = True
except ImportError:
    print("⚠️ OCRBench evaluation modules not available - basic evaluation will be used")
    OCRBENCH_EVAL_AVAILABLE = False


class OCRBenchmarkInfer(BenchmarkInfer):
    """
    OCRBench specific implementation of the BenchmarkInfer base class.

    Supports various OCR tasks with proper evaluation metrics.
    """

    def __init__(self):
        """Initialize OCRBench inference system."""
        super().__init__(
            benchmark_name="ocrbench",
            image_cache_dir=None  # Use default cache location
        )

        # OCRBench specific settings
        self.skip_complex_types = True  # Skip memory-intensive question types by default
        self.complex_types = [
            "key information extraction cn", "key information extraction en",
            "key information mapping en", "chart parsing en", "document parsing cn",
            "document parsing en", "handwritten answer extraction cn",
            "table parsing cn", "table parsing en"
        ]

    def set_skip_complex_types(self, skip: bool):
        """Set whether to skip complex question types that may cause memory issues."""
        self.skip_complex_types = skip

    # ==================== Abstract Method Implementations ====================

    def build_flow_input(self, sample: Dict[str, Any]) -> str:
        """
        Convert OCRBench sample to flow input format.

        Args:
            sample: OCRBench sample dict with question, image, and metadata

        Returns:
            Formatted flow input string
        """
        question = sample.get('question', '')
        question_type = sample.get('type', 'basic')

        # Add task-specific context based on question type
        task_context = self._get_task_context(question_type)

        # Format the question with context
        if task_context:
            formatted_question = f"{task_context}\n\nQuestion: {question}"
        else:
            formatted_question = question

        # Handle image - try different possible image fields
        image_path = None
        if 'image_path' in sample and sample['image_path']:
            image_path = sample['image_path']
        elif 'image_base64' in sample:
            # Convert base64 to cached image
            try:
                image_path = self._save_base64_image_to_cache(
                    sample['image_base64'],
                    sample.get('id', 'unknown')
                )
            except Exception as e:
                print(f"⚠️ Failed to process base64 image for sample {sample.get('id', '')}: {e}")
        elif 'image' in sample and hasattr(sample['image'], 'save'):
            # PIL Image object
            try:
                image_path = self._save_pil_image_to_cache(
                    sample['image'],
                    sample.get('id', 'unknown')
                )
            except Exception as e:
                print(f"⚠️ Failed to process PIL image for sample {sample.get('id', '')}: {e}")

        # Combine question and image
        if image_path:
            return f"{formatted_question}\nimage_path:{image_path}"
        else:
            return formatted_question

    def _get_task_context(self, question_type: str) -> str:
        """Get task-specific context for different OCR question types."""
        context_map = {
            "text recognition en": "Please read and transcribe all text visible in the image accurately.",
            "text recognition cn": "请准确阅读并转录图像中的所有可见文字。",
            "formula recognition en": "Please recognize and transcribe the mathematical formula(s) in the image using LaTeX notation.",
            "formula recognition cn": "请识别并使用LaTeX记号转录图像中的数学公式。",
            "table parsing en": "Please parse the table structure and content from the image. Provide the result in the requested format (HTML or Markdown).",
            "table parsing cn": "请解析图像中的表格结构和内容，以HTML格式提供结果。",
            "document parsing en": "Please extract and organize the document content from the image.",
            "document parsing cn": "请从图像中提取并整理文档内容。",
            "key information extraction en": "Please extract key information from the image as requested in the question.",
            "key information extraction cn": "请根据问题要求从图像中提取关键信息。",
            "text spotting en": "Please locate and identify text in the image, providing bounding box coordinates if requested.",
            "text grounding en": "Please locate the specified text in the image and provide its position.",
            "chart parsing en": "Please analyze and extract data from the chart/graph in the image.",
            "full-page OCR en": "Please perform full-page OCR to extract all text content from the image.",
            "full-page OCR cn": "请对图像进行全页面OCR，提取所有文字内容。"
        }
        return context_map.get(question_type, "")

    def _save_base64_image_to_cache(self, base64_str: str, sample_id: str) -> str:
        """Save base64 image string to cache directory."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for image processing")

        # Decode base64 to PIL Image
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))

        return self._save_pil_image_to_cache(image, sample_id)

    def _save_pil_image_to_cache(self, image: Image.Image, sample_id: str) -> str:
        """Save PIL Image to cache directory."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for image processing")

        # Generate cache filename
        import hashlib
        cache_name = hashlib.md5(f"ocrbench_{sample_id}".encode()).hexdigest() + ".jpg"
        cache_path = self.image_cache_dir / cache_name

        # Check if already cached
        if cache_path.exists():
            return str(cache_path)

        try:
            # Convert to RGB and save
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large
            max_size = (2048, 2048)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

            image.save(cache_path, format='JPEG', quality=85)
            return str(cache_path)

        except Exception as e:
            print(f"❌ Error processing image for {sample_id}: {e}")
            raise

    def extract_question(self, sample: Dict[str, Any]) -> str:
        """Extract question text from OCRBench sample."""
        return sample.get('question', '')

    def extract_options(self, sample: Dict[str, Any]) -> List[str]:
        """Extract answer options from OCRBench sample (most are open-ended)."""
        # OCRBench typically doesn't have multiple choice options
        # But some tasks might have specific answer formats
        return []

    def extract_expected_answer(self, sample: Dict[str, Any]) -> str:
        """Extract expected answer from OCRBench sample."""
        answers = sample.get('answers', [])
        if answers and isinstance(answers, list):
            return answers[0] if len(answers) > 0 else ""
        return str(answers) if answers else ""

    def extract_question_id(self, sample: Dict[str, Any]) -> str:
        """Extract unique question ID from OCRBench sample."""
        # Use current_index if available (for sequential naming)
        if 'current_index' in sample:
            return f"question_{sample['current_index']}"
        # Fallback to sample ID
        return str(sample.get('id', 'unknown'))

    def extract_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from OCRBench sample for logging."""
        return {
            'ocr_sample_id': sample.get('id', ''),
            'question_type': sample.get('type', ''),
            'dataset_name': sample.get('dataset_name', ''),
            'eval_method': sample.get('eval', ''),
            'current_index': sample.get('current_index', ''),
            'benchmark': 'ocrbench'
        }

    def stream_json_samples(self, json_file: str, limit: int = -1):
        """Stream JSON samples one by one to avoid memory overflow."""
        print(f"📂 Streaming JSON samples from: {json_file}")

        count = 0
        try:
            # First, try to load as a complete JSON object (like the original qwen7b script)
            with open(json_file, 'r', encoding='utf-8') as f:
                print("📄 Loading complete JSON file to extract samples array")
                data = json.load(f)

                # Extract samples from the data structure
                if isinstance(data, dict) and 'samples' in data:
                    samples = data['samples']
                    print(f"📊 Found {len(samples)} samples in JSON object")
                elif isinstance(data, list):
                    samples = data
                    print(f"📊 Found {len(samples)} samples in JSON array")
                else:
                    print("❌ Unexpected JSON structure - expected object with 'samples' key or array")
                    return

                # Stream the samples one by one
                for sample in samples:
                    if limit > 0 and count >= limit:
                        break
                    count += 1
                    yield sample

        except json.JSONDecodeError as e:
            # Fallback to JSONL format (one JSON per line)
            print(f"⚠️ Failed to parse as complete JSON object: {e}")
            print("📄 Falling back to JSONL format (one JSON per line)")

            count = 0
            with open(json_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if limit > 0 and count >= limit:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        sample = json.loads(line)
                        count += 1
                        yield sample
                    except json.JSONDecodeError as line_error:
                        print(f"⚠️ Skipping line {line_num}, JSON decode error: {line_error}")
                        continue

        print(f"✅ Streaming completed, processed {count} samples")

    def should_skip_complex_type(self, question_type: str) -> bool:
        """Check if question type should be skipped due to complexity."""
        if not self.skip_complex_types:
            return False
        return question_type in self.complex_types

    def base64_to_data_url(self, b64_string: str, fmt: str = "PNG") -> str:
        """Convert base64 string to data URL."""
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{b64_string}"

    def process_sample_with_flow(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Process a single sample using the flow system.

        Args:
            sample: OCRBench sample dictionary
            index: Current sample index

        Returns:
            Processing result dictionary
        """
        # Ensure sample is a dictionary
        if isinstance(sample, str):
            try:
                sample = json.loads(sample)
            except json.JSONDecodeError:
                return {
                    'success': False,
                    'skipped': False,
                    'error': 'Invalid JSON format in sample',
                    'execution_time': 0,
                    'sample_id': 'unknown',
                    'question_type': 'unknown'
                }

        # Make a copy to avoid modifying the original
        sample = sample.copy() if isinstance(sample, dict) else {}

        # Add current index for sequential naming
        sample['current_index'] = index

        # Extract sample information
        question_type = sample.get("type", "basic")
        sample_id = sample.get("id", index)

        # Check if should skip complex types
        if self.should_skip_complex_type(question_type):
            return {
                'success': False,
                'skipped': True,
                'reason': f"Skipped complex question type: {question_type}",
                'sample_id': sample_id,
                'question_type': question_type
            }

        try:
            # Record start time
            start_time = datetime.now()

            # Execute flow
            response = self.execute_on_sample(sample)

            # Record end time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            return {
                'success': True,
                'skipped': False,
                'response': response,
                'execution_time': execution_time,
                'sample_id': sample_id,
                'question_type': question_type,
                'start_time': start_time,
                'end_time': end_time
            }

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() if 'start_time' in locals() else 0

            return {
                'success': False,
                'skipped': False,
                'error': str(e),
                'execution_time': execution_time,
                'sample_id': sample_id,
                'question_type': question_type
            }

    def process_streaming_samples(self, json_file: str, limit: int = -1, verbose: bool = False) -> Dict[str, Any]:
        """
        Process samples using streaming approach to avoid memory overflow.

        Args:
            json_file: Path to JSON file with samples
            limit: Maximum number of samples to process (-1 for all)
            verbose: Whether to print detailed execution info

        Returns:
            Processing statistics
        """
        print("🔄 Using streaming processing mode to avoid memory overflow")

        # Initialize counters
        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Stream samples and process them
        sample_iter = self.stream_json_samples(json_file, limit)
        progress_bar = tqdm(sample_iter, desc="Streaming OCR Inference")

        for i, sample in enumerate(progress_bar):
            # Process sample with flow
            result = self.process_sample_with_flow(sample, i)

            if result['skipped']:
                print(f"⏭️ Skipped complex question type: {result['question_type']} (ID: {result['sample_id']})")
                skipped_count += 1
                continue
            elif not result['success']:
                print(f"❌ Error processing sample {result['sample_id']}: {result.get('error', 'Unknown error')}")
                error_count += 1
                continue
            else:
                processed_count += 1

                # Print execution summary if verbose
                if verbose:
                    self.print_execution_summary(
                        i, limit if limit > 0 else "streaming",
                        result['response'], result['execution_time'], sample
                    )

                # Print completion status
                print(f"✅ Question {i}/{limit if limit > 0 else '?'} completed | "
                     f"Type: {result['question_type']} | Time: {result['execution_time']:.2f}s")

        return {
            'processed': processed_count,
            'skipped': skipped_count,
            'errors': error_count,
            'total_attempted': processed_count + skipped_count + error_count
        }

    def load_dataset(self, data_source: str, **kwargs) -> pd.DataFrame:
        """
        Load OCRBench dataset from various sources.

        Args:
            data_source: Path to JSON file or parquet directory
            **kwargs: Additional arguments (max_samples, samples_per_type, etc.)

        Returns:
            pandas DataFrame with OCRBench samples
        """
        max_samples = kwargs.get('max_samples')
        samples_per_type = kwargs.get('samples_per_type', 30)
        max_types = kwargs.get('max_types', 30)

        if data_source.endswith('.json') and os.path.exists(data_source):
            # Load from JSON file
            print(f"📂 Loading OCRBench data from JSON: {data_source}")
            return self._load_from_json(data_source, max_samples)

        elif os.path.isdir(data_source) and DATASETS_AVAILABLE:
            # Load from parquet files
            print(f"📂 Loading OCRBench data from parquet: {data_source}")
            return self._load_from_parquet(data_source, samples_per_type, max_types, max_samples)

        else:
            raise FileNotFoundError(f"OCRBench data not found at {data_source}")

    def _load_from_json(self, json_file: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Load OCRBench data from JSON file (following qwen7b script pattern)."""
        print(f"📂 Loading OCRBench data from JSON file: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract samples and metadata (following original qwen7b pattern)
        if isinstance(data, dict):
            if 'samples' in data:
                samples = data['samples']
                metadata = data.get('metadata', {})

                print(f"✅ Loading completed: {len(samples)} samples")
                if 'question_types' in metadata:
                    print(f"📊 Question types: {len(metadata['question_types'])}")
                if 'dataset_names' in metadata:
                    print(f"📋 Datasets: {len(metadata['dataset_names'])}")
            else:
                # Assume the entire dict is samples data
                samples = [data]
                print(f"📊 Single sample loaded")
        elif isinstance(data, list):
            samples = data
            print(f"📊 Array of {len(samples)} samples loaded")
        else:
            raise ValueError(f"Unsupported JSON format in {json_file}. Expected dict with 'samples' key or list.")

        # Limit samples if requested
        if max_samples and max_samples > 0:
            original_count = len(samples)
            samples = samples[:max_samples]
            print(f"📊 Limited to {len(samples)} samples (from {original_count} total)")

        df = pd.DataFrame(samples)
        print(f"📊 Created DataFrame with {len(df)} samples")

        return df

    def _load_from_parquet(self, data_dir: str, samples_per_type: int, max_types: int, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Load OCRBench data from parquet files with sampling."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required for parquet loading")

        ds = load_dataset("parquet", data_files=os.path.join(data_dir, "*.parquet"), split="train")

        # Sample by question type
        if samples_per_type > 0:
            print(f"🎯 Sampling by question type: {samples_per_type} per type, max {max_types} types")

            # Group by type
            type_groups = {}
            for i, sample in enumerate(ds):
                q_type = sample.get('type', 'unknown')
                if q_type not in type_groups:
                    type_groups[q_type] = []
                type_groups[q_type].append(i)

            # Select samples
            selected_indices = []
            selected_types = list(type_groups.keys())[:max_types]

            for q_type in selected_types:
                indices = type_groups[q_type][:samples_per_type]
                selected_indices.extend(indices)

            # Create DataFrame from selected samples
            selected_samples = []
            for idx in selected_indices:
                sample = ds[idx]
                # Convert PIL image to dict for DataFrame storage
                sample_dict = {
                    'id': sample.get('id', idx),
                    'question': sample.get('question', ''),
                    'answers': sample.get('answers', []),
                    'type': sample.get('type', ''),
                    'dataset_name': sample.get('dataset_name', ''),
                    'eval': sample.get('eval', ''),
                    'image': sample.get('image')  # Keep PIL image object
                }
                selected_samples.append(sample_dict)

            df = pd.DataFrame(selected_samples)
        else:
            # Convert entire dataset
            samples = list(ds)
            if max_samples:
                samples = samples[:max_samples]
            df = pd.DataFrame(samples)

        print(f"📊 Loaded OCRBench parquet dataset: {len(df)} samples")
        return df

    def should_skip_sample(self, sample: Dict[str, Any]) -> bool:
        """Check if sample should be skipped based on question type."""
        if not self.skip_complex_types:
            return False

        question_type = sample.get('type', '')
        return question_type in self.complex_types

    def print_sample_info(self, sample: Dict[str, Any]):
        """Print OCRBench specific sample information."""
        print(f"ID: {sample.get('id', 'N/A')}")
        print(f"Type: {sample.get('type', 'N/A')}")
        print(f"Dataset: {sample.get('dataset_name', 'N/A')}")
        print(f"Eval Method: {sample.get('eval', 'N/A')}")

        question = sample.get('question', '')
        if len(question) > 100:
            print(f"Question: {question[:100]}...")
        else:
            print(f"Question: {question}")

        expected = sample.get('answers', [])
        if isinstance(expected, list) and expected:
            expected_str = str(expected[0])
            if len(expected_str) > 50:
                print(f"Expected: {expected_str[:50]}...")
            else:
                print(f"Expected: {expected_str}")


def run_flow_inference(args):
    """Main inference pipeline for OCRBench."""
    print("🚀 Starting OCRBench evaluation with IterativeRefinementFlow")
    print("="*70)

    # Setup vLLM if needed
    if not args.skip_vllm_setup:
        try:
            from app.utils.vllm_setup import check_and_setup_vllm
            check_and_setup_vllm(vision_port=args.vision_port, text_port=args.text_port)
        except ImportError:
            print("⚠️ vLLM setup not available - proceeding without setup")

    # Initialize OCRBench benchmark inference system
    ocr_infer = OCRBenchmarkInfer()
    print(f"🔄 Max iterations: {ocr_infer.flow_executor.underlying_flow.max_iterations}")

    # Configure complex type handling
    ocr_infer.set_skip_complex_types(args.skip_complex_types)
    if args.skip_complex_types:
        print("⚠️ Skipping complex question types to avoid memory issues")

    # Build experiment configuration
    experiment_config = ocr_infer.build_experiment_config(args)
    experiment_config.update({
        'data_source': args.data_source,
        'skip_complex_types': args.skip_complex_types,
        'samples_per_type': getattr(args, 'samples_per_type', 30),
        'max_types': getattr(args, 'max_types', 30),
        'streaming': getattr(args, 'streaming', False)
    })

    # Start logging session
    ocr_infer.start_logging_session(experiment_config)

    print(f"📁 Session directory: {ocr_infer.session_dir}")
    print(f"📄 Individual question results will be saved as JSON files")

    # Choose processing mode: streaming or batch
    if (getattr(args, 'streaming', False) and
        args.data_source.endswith('.json') and
        os.path.exists(args.data_source)):

        # Check if start/end sample ranges are requested
        if args.start_sample is not None or args.end_sample is not None:
            print("⚠️ Warning: --start-sample and --end-sample are not supported in streaming mode")
            print("💡 Switching to batch processing mode to support sample range filtering")

            # Fall back to batch mode for range filtering
            print("📊 Loading OCRBench dataset for batch processing (range filtering required)...")
            data = ocr_infer.load_dataset(
                args.data_source,
                max_samples=args.max_samples,
                samples_per_type=getattr(args, 'samples_per_type', 30),
                max_types=getattr(args, 'max_types', 30)
            )

            # Filter dataset based on arguments
            data = ocr_infer.filter_dataset(
                data,
                start_sample=args.start_sample,
                end_sample=args.end_sample,
                max_samples=args.max_samples
            )

            # Update experiment config
            experiment_config['total_samples'] = len(data)
            streaming_mode = False
        else:
            # Streaming mode for large JSON files
            print("🎯 Using streaming processing mode for large datasets")

            # Update experiment config with estimated total
            limit = args.max_samples if args.max_samples else -1
            experiment_config['total_samples'] = limit if limit > 0 else 'streaming'
            streaming_mode = True

        if streaming_mode:
            # Process with streaming
            stats = ocr_infer.process_streaming_samples(
                args.data_source,
                limit=limit,
                verbose=args.verbose
            )

            processed_count = stats['processed']
            skipped_count = stats['skipped']
            error_count = stats['errors']

            print(f"✅ Streaming OCRBench inference completed!")
            print(f"📊 Processed: {processed_count} samples")
            print(f"⏭️ Skipped: {skipped_count} samples")
            print(f"❌ Errors: {error_count} samples")
        else:
            # Batch processing (fell back from streaming due to range filtering)
            print(f"🔄 Processing {len(data)} samples in batch mode...")
            processed_count = 0
            skipped_count = 0

            for i in tqdm(range(len(data)), desc="Running OCRBench inference"):
                sample = data.iloc[i].to_dict()

                # Get the original DataFrame index to preserve filtering
                original_index = data.index[i]

                # Process sample using the unified method with original index
                result = ocr_infer.process_sample_with_flow(sample, original_index)

                if result['skipped']:
                    print(f"⏭️ Skipping complex question type: {result['question_type']} (ID: {result['sample_id']})")
                    skipped_count += 1
                    continue
                elif not result['success']:
                    print(f"❌ Error processing sample {result['sample_id']}: {result.get('error', 'Unknown error')}")
                    continue
                else:
                    processed_count += 1

                    # Print execution summary if verbose
                    if args.verbose:
                        ocr_infer.print_execution_summary(
                            i, len(data), result['response'], result['execution_time'], sample
                        )

                    # Print completion status
                    print(f"✅ Question {i}/{len(data)} completed | "
                         f"Type: {result['question_type']} | Time: {result['execution_time']:.2f}s")

            print(f"✅ OCRBench batch inference completed!")
            print(f"📊 Processed: {processed_count} samples")
            print(f"⏭️ Skipped: {skipped_count} samples")

    else:
        # Traditional batch processing mode
        print("📊 Loading OCRBench dataset for batch processing...")
        data = ocr_infer.load_dataset(
            args.data_source,
            max_samples=args.max_samples,
            samples_per_type=getattr(args, 'samples_per_type', 30),
            max_types=getattr(args, 'max_types', 30)
        )

        # Filter dataset based on arguments
        data = ocr_infer.filter_dataset(
            data,
            start_sample=args.start_sample,
            end_sample=args.end_sample,
            max_samples=args.max_samples
        )

        # Update experiment config with actual sample count
        experiment_config['total_samples'] = len(data)

        # Process samples in batch mode
        print(f"🔄 Processing {len(data)} samples in batch mode...")
        processed_count = 0
        skipped_count = 0

        for i in tqdm(range(len(data)), desc="Running OCRBench inference"):
            sample = data.iloc[i].to_dict()

            # Get the original DataFrame index to preserve filtering
            original_index = data.index[i]

            # Process sample using the unified method with original index
            result = ocr_infer.process_sample_with_flow(sample, original_index)

            if result['skipped']:
                print(f"⏭️ Skipping complex question type: {result['question_type']} (ID: {result['sample_id']})")
                skipped_count += 1
                continue
            elif not result['success']:
                print(f"❌ Error processing sample {result['sample_id']}: {result.get('error', 'Unknown error')}")
                continue
            else:
                processed_count += 1

                # Print execution summary if verbose
                if args.verbose:
                    ocr_infer.print_execution_summary(
                        i, len(data), result['response'], result['execution_time'], sample
                    )

                # Print completion status
                print(f"✅ Question {i}/{len(data)} completed | "
                     f"Type: {result['question_type']} | Time: {result['execution_time']:.2f}s")

        print(f"✅ OCRBench batch inference completed!")
        print(f"📊 Processed: {processed_count} samples")
        print(f"⏭️ Skipped: {skipped_count} samples")

    # Save session directory reference
    session_directory = ocr_infer.session_dir

    # Finish logging session
    ocr_infer.finish_logging_session()

    # Generate summary files
    print("\n📄 Generating summary files...")
    try:
        utils_path = current_dir.parent / "utils"
        sys.path.insert(0, str(utils_path))

        from result_converter import (
            generate_summary_csv_from_questions,
            generate_evaluation_jsonl_from_questions
        )

        if session_directory:
            csv_output_file = session_directory / "inference_summary.csv"
            jsonl_output_file = session_directory / "inference_results.jsonl"

            # For streaming mode, we need to get data differently
            if getattr(args, 'streaming', False):
                print("⚠️ Summary generation limited in streaming mode")
                # Could implement a lightweight summary here if needed
            else:
                generate_summary_csv_from_questions(session_directory, data, csv_output_file, "ocrbench")
                generate_evaluation_jsonl_from_questions(session_directory, data, jsonl_output_file, "ocrbench")

                print(f"✅ Summary CSV saved to {csv_output_file}")
                print(f"✅ Evaluation JSONL saved to {jsonl_output_file}")

    except ImportError:
        print("⚠️ Result conversion utilities not available")

    print(f"\n✅ OCRBench evaluation completed!")
    if session_directory:
        print(f"📁 Session directory: {session_directory}")

        # Save session info
        ocr_infer.save_session_info()

    return session_directory


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="OCRBench Evaluation with IterativeRefinementFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference from JSON file
  python infer_ocrbench_flow.py --data-source ocrbench_local_data.json --max-samples 50

  # Streaming mode for large JSON files (avoids memory overflow)
  python infer_ocrbench_flow.py --data-source large_ocrbench_data.json --streaming --max-samples 1000

  # Inference from parquet files with sampling
  python infer_ocrbench_flow.py --data-source OCRBench-v2/data --samples-per-type 10 --max-types 5

  # Process specific sample range
  python infer_ocrbench_flow.py --data-source data.json --start-sample 100 --end-sample 200

  # Include complex question types (may cause memory issues)
  python infer_ocrbench_flow.py --data-source data.json --include-complex-types

  # Streaming with complex types included
  python infer_ocrbench_flow.py --data-source data.json --streaming --include-complex-types --verbose
        """
    )

    # Data configuration
    parser.add_argument("--data-source", type=str, required=True,
                       help="Path to OCRBench JSON file or parquet directory")

    # Sample range configuration
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--start-sample", type=int, help="Starting sample index (0-based)")
    parser.add_argument("--end-sample", type=int, help="Ending sample index (inclusive)")

    # Sampling configuration for parquet data
    parser.add_argument("--samples-per-type", type=int, default=30,
                       help="Number of samples per question type (for parquet data)")
    parser.add_argument("--max-types", type=int, default=30,
                       help="Maximum number of question types to process")

    # Question type filtering
    parser.add_argument("--include-complex-types", action="store_true",
                       help="Include complex question types (may cause memory issues)")

    # Processing mode
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming processing mode to avoid memory overflow (for JSON files)")

    # System configuration
    parser.add_argument("--skip-vllm-setup", action="store_true",
                       help="Skip automatic vLLM cluster setup")
    parser.add_argument("--vision-port", type=int, default=8000,
                       help="Port for vision model API")
    parser.add_argument("--text-port", type=int, default=8001,
                       help="Port for text model API")

    # Output configuration
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed execution info")

    args = parser.parse_args()

    # Set complex types flag
    args.skip_complex_types = not args.include_complex_types

    # Validate arguments
    if args.start_sample is not None and args.end_sample is not None:
        if args.start_sample >= args.end_sample:
            print("❌ Error: start-sample must be less than end-sample")
            return

    if args.max_samples is not None and (args.start_sample is not None or args.end_sample is not None):
        print("❌ Error: Cannot use --max-samples with --start-sample/--end-sample")
        return

    # Check data source exists
    if not os.path.exists(args.data_source):
        print(f"❌ Data source not found: {args.data_source}")
        return

    # Run inference
    session_dir = run_flow_inference(args)
    if session_dir:
        print(f"\n🎯 OCRBench evaluation completed successfully!")
        print(f"📁 Session directory: {session_dir}")


if __name__ == "__main__":
    main()
