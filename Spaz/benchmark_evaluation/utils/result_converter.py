#!/usr/bin/env python3
"""
Utility functions for converting MMMU flow results between different formats.

This module provides functions to:
1. Generate summary CSV files from individual question JSON files
2. Generate evaluation-compatible JSONL files from question JSON files
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def generate_summary_csv_from_questions(session_dir: Path, original_data: pd.DataFrame, output_file: Path, task: str):
    """
    Generate a summary CSV by reading through all question_{id}.json files in the session directory.

    Args:
        session_dir: Path to the session directory containing question JSON files
        original_data: Original dataset DataFrame for additional fields (works with MMMU, MIA, etc.)
        output_file: Path to save the summary CSV
        task: Task name (e.g., 'MMMU_DEV_VAL', 'mia')
    """
    summary_rows = []

    # Create a mapping from question index to original data
    # Handle different dataset schemas (some have 'index', some don't)
    original_data_map = {}
    for i, row in original_data.iterrows():
        # Use 'index' column if available, otherwise use DataFrame index
        index_key = row.get('index', i) if 'index' in row else i
        original_data_map[index_key] = row.to_dict()

    # Find all question JSON files.
    # Files are named after question_id (e.g. validation_0.json, dev_0.json, question_0.json)
    question_files = []
    for pattern in ["validation_*.json", "dev_*.json", "question_*.json"]:
        question_files.extend(session_dir.glob(pattern))
    # Exclude session_metadata.json and other non-question files
    question_files = sorted(
        set(f for f in question_files if f.name != "session_metadata.json"),
        key=lambda p: p.name
    )

    for question_file in question_files:
        try:
            with open(question_file, 'r', encoding='utf-8') as f:
                question_data = json.load(f)

            # Extract question number from question_id
            # Handles: validation_0, dev_0, question_0
            question_id = question_data['question_id']
            try:
                question_num = int(question_id.split('_')[-1])
            except (ValueError, IndexError):
                question_num = 0

            # Get original data for this question
            metadata = question_data.get('metadata', {})
            # Try different possible index fields
            original_index = metadata.get('full_dataset_index', metadata.get('index', question_num))
            original_row = original_data_map.get(original_index, {})

            # Count messages
            messages_count = len(question_data.get('messages', []))

            # Check if has image
            has_image = bool(original_row.get('image') and str(original_row.get('image')) != 'nan')

            # Build summary row with all requested fields
            summary_row = {
                'question_id': question_data['question_id'],
                'task': task,
                'id': original_row.get('id', ''),
                'index': original_row.get('index', original_index),
                'question': question_data.get('raw_question', original_row.get('question', '')),
                'split': original_row.get('split', ''),
                'A': original_row.get('A', ''),
                'B': original_row.get('B', ''),
                'C': original_row.get('C', ''),
                'D': original_row.get('D', ''),
                'answer': question_data.get('expected_answer', original_row.get('answer', '')),
                'topic_difficulty': original_row.get('topic_difficulty', ''),
                'subfield': original_row.get('subfield', ''),
                'image_type': original_row.get('image_type', ''),
                'question_type': original_row.get('question_type', ''),
                'explanation': original_row.get('explanation', ''),
                'generated_response': question_data.get('model_response', ''),
                'has_image': has_image,
                'messages_count': messages_count,
                # Additional useful fields
                'duration_seconds': question_data.get('duration_seconds', 0),
                'category': metadata.get('category', original_row.get('category', '')),
                'total_tokens': question_data.get('token_usage', {}).get('total_tokens', 0),
                'total_input_tokens': question_data.get('token_usage', {}).get('total_input_tokens', 0),
                'total_completion_tokens': question_data.get('token_usage', {}).get('total_completion_tokens', 0),
            }

            summary_rows.append(summary_row)

        except Exception as e:
            print(f"⚠️ Warning: Failed to process {question_file}: {e}")

    # Convert to DataFrame and save as CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        # Sort by question_id for consistency
        summary_df = summary_df.sort_values('question_id')

        # Save to CSV
        summary_df.to_csv(output_file, index=False)
        print(f"📊 Generated summary with {len(summary_rows)} questions")

        # Print some statistics
        if 'generated_response' in summary_df.columns and 'answer' in summary_df.columns:
            # Simple accuracy calculation (exact match)
            correct_responses = 0
            total_responses = 0

            for _, row in summary_df.iterrows():
                if row['answer'] and row['generated_response']:
                    total_responses += 1
                    # Extract first letter from generated response for comparison
                    generated_first_char = row['generated_response'].strip()[:1].upper()
                    expected_answer = str(row['answer']).strip().upper()

                    if generated_first_char == expected_answer:
                        correct_responses += 1

            if total_responses > 0:
                accuracy = correct_responses / total_responses * 100
                print(f"📈 Simple accuracy: {correct_responses}/{total_responses} ({accuracy:.1f}%)")

        # Token usage statistics
        if 'total_tokens' in summary_df.columns:
            total_tokens = summary_df['total_tokens'].sum()
            avg_tokens_per_question = summary_df['total_tokens'].mean()
            print(f"🎯 Token usage: {total_tokens:,} total tokens, {avg_tokens_per_question:.0f} avg per question")
    else:
        print("⚠️ No question data found to generate summary")


def generate_evaluation_jsonl_from_questions(session_dir: Path, original_data: pd.DataFrame, output_file: Path, task: str):
    """
    Generate evaluation-compatible JSONL file from individual question JSON files.
    This creates the format expected by the existing run_evaluation function.

    Args:
        session_dir: Path to the session directory containing question JSON files
        original_data: Original dataset DataFrame (works with MMMU, MIA, etc.)
        output_file: Path to save the JSONL file
        task: Task name (e.g., 'MMMU_DEV_VAL', 'mia')
    """
    # Create a mapping from question index to original data
    # Handle different dataset schemas (some have 'index', some don't)
    original_data_map = {}
    for i, row in original_data.iterrows():
        # Use 'index' column if available, otherwise use DataFrame index
        index_key = row.get('index', i) if 'index' in row else i
        original_data_map[index_key] = row.to_dict()

    # Find all question JSON files.
    # Files are named after question_id (e.g. validation_0.json, dev_0.json, question_0.json)
    question_files = []
    for pattern in ["validation_*.json", "dev_*.json", "question_*.json"]:
        question_files.extend(session_dir.glob(pattern))
    question_files = sorted(
        set(f for f in question_files if f.name != "session_metadata.json"),
        key=lambda p: p.name
    )

    results = []
    for question_file in question_files:
        try:
            with open(question_file, 'r', encoding='utf-8') as f:
                question_data = json.load(f)

            # Extract question number from question_id
            # Handles: validation_0, dev_0, question_0
            question_id = question_data['question_id']
            try:
                question_num = int(question_id.split('_')[-1])
            except (ValueError, IndexError):
                question_num = 0

            # Get original data for this question
            metadata = question_data.get('metadata', {})
            # Try different possible index fields
            original_index = metadata.get('full_dataset_index', metadata.get('index', question_num))
            original_row = original_data_map.get(original_index, {})

            # Clean original row to remove image data
            cleaned_original_row = original_row.copy()
            if 'image' in cleaned_original_row:
                # Keep only a marker that image was present, not the actual data
                cleaned_original_row['image'] = "[IMAGE_DATA_EXCLUDED]" if original_row.get('image') else None

            # Create evaluation-compatible format
            eval_result = {
                "question_id": original_index,
                "annotation": cleaned_original_row,  # Cleaned original dataset row
                "task": task,
                "result": {
                    "gen": question_data.get('model_response', '')
                },
                "flow_metadata": {
                    "question_id": question_data['question_id'],
                    "duration_seconds": question_data.get('duration_seconds', 0),
                    "total_tokens": question_data.get('token_usage', {}).get('total_tokens', 0)
                }
            }

            results.append(eval_result)

        except Exception as e:
            print(f"⚠️ Warning: Failed to process {question_file} for evaluation: {e}")

    # Write JSONL file
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"📄 Generated evaluation JSONL with {len(results)} questions")
    else:
        print("⚠️ No question data found to generate evaluation JSONL")


def convert_session_results(session_dir: Path, original_data_path: Path, task: str = "MMMU_DEV_VAL",
                          generate_csv: bool = True, generate_jsonl: bool = True):
    """
    Convenience function to convert a session directory to both CSV and JSONL formats.

    Args:
        session_dir: Path to the session directory containing question JSON files
        original_data_path: Path to the original MMMU dataset file (TSV/CSV)
        task: Task name (e.g., 'MMMU_DEV_VAL')
        generate_csv: Whether to generate summary CSV
        generate_jsonl: Whether to generate evaluation JSONL
    """
    import pandas as pd

    print(f"🔄 Converting session results from: {session_dir}")

    # Load original data
    if original_data_path.suffix.lower() == '.tsv':
        original_data = pd.read_csv(original_data_path, sep='\t')
    else:
        original_data = pd.read_csv(original_data_path)

    print(f"📊 Loaded original data with {len(original_data)} rows")

    # Generate outputs
    if generate_csv:
        csv_output = session_dir / "summary.csv"
        generate_summary_csv_from_questions(session_dir, original_data, csv_output, task)
        print(f"✅ CSV summary saved to: {csv_output}")

    if generate_jsonl:
        jsonl_output = session_dir / "inference_results.jsonl"
        generate_evaluation_jsonl_from_questions(session_dir, original_data, jsonl_output, task)
        print(f"✅ JSONL inference file saved to: {jsonl_output}")


if __name__ == "__main__":
    """
    Command-line interface for converting session results.

    Usage:
        python result_converter.py /path/to/session/dir /path/to/original/data.tsv [task_name]
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python result_converter.py <session_dir> <original_data_path> [task_name]")
        print("Example: python result_converter.py logs/mmmu/session_20250921_141131_293 data/MMMU/MMMU_DEV_VAL.tsv MMMU_DEV_VAL")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    original_data_path = Path(sys.argv[2])
    task = sys.argv[3] if len(sys.argv) > 3 else "MMMU_DEV_VAL"

    if not session_dir.exists():
        print(f"❌ Error: Session directory not found: {session_dir}")
        sys.exit(1)

    if not original_data_path.exists():
        print(f"❌ Error: Original data file not found: {original_data_path}")
        sys.exit(1)

    convert_session_results(session_dir, original_data_path, task)
    print("🎉 Conversion completed!")