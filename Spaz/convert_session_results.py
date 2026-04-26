#!/usr/bin/env python3
"""
Standalone script to convert MMMU session results to CSV and JSONL formats.

This script can be used independently to convert existing session directories
containing question_*.json, validation_*.json, or dev_*.json files into summary CSV and evaluation JSONL formats.

Usage Examples:
    # Convert a single session directory
    python convert_session_results.py logs/mmmu/session_20250921_141131_293 data/MMMU_DEV_VAL.tsv

    # Convert with custom task name
    python convert_session_results.py logs/mmmu/session_20250921_141131_293 data/MMMU_DEV_VAL.tsv --task MMMU_DEV_VAL

    # Generate only CSV (skip JSONL)
    python convert_session_results.py logs/mmmu/session_20250921_141131_293 data/MMMU_DEV_VAL.tsv --no-jsonl

    # Generate only JSONL (skip CSV)
    python convert_session_results.py logs/mmmu/session_20250921_141131_293 data/MMMU_DEV_VAL.tsv --no-csv
"""

import sys
import argparse
from pathlib import Path

# Add the utils directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "benchmark_evaluation" / "utils"))

from result_converter import convert_session_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert MMMU session results to CSV and JSONL formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "session_dir",
        type=Path,
        help="Path to the session directory containing question_*.json, validation_*.json, or dev_*.json files"
    )

    parser.add_argument(
        "original_data_path",
        type=Path,
        help="Path to the original MMMU dataset file (TSV/CSV format)"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="MMMU_DEV_VAL",
        help="Task name for the output files (default: MMMU_DEV_VAL)"
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip generating CSV summary file"
    )

    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Skip generating JSONL evaluation file"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.session_dir.exists():
        print(f"❌ Error: Session directory not found: {args.session_dir}")
        sys.exit(1)

    if not args.original_data_path.exists():
        print(f"❌ Error: Original data file not found: {args.original_data_path}")
        sys.exit(1)

    # Check if session directory contains question files (support all patterns)
    question_files = list(args.session_dir.glob("question_*.json"))
    validation_files = list(args.session_dir.glob("validation_*.json"))
    dev_files = list(args.session_dir.glob("dev_*.json"))

    all_files = question_files + validation_files + dev_files

    if not all_files:
        print(f"❌ Error: No question_*.json, validation_*.json, or dev_*.json files found in {args.session_dir}")
        sys.exit(1)

    # Report what was found
    file_types = []
    if question_files:
        file_types.append(f"{len(question_files)} question_*.json")
    if validation_files:
        file_types.append(f"{len(validation_files)} validation_*.json")
    if dev_files:
        file_types.append(f"{len(dev_files)} dev_*.json")

    print(f"📁 Found {' and '.join(file_types)} files in session directory")

    # Convert results
    try:
        convert_session_results(
            session_dir=args.session_dir,
            original_data_path=args.original_data_path,
            task=args.task,
            generate_csv=not args.no_csv,
            generate_jsonl=not args.no_jsonl
        )
        print("\n🎉 Conversion completed successfully!")

        # Show output files
        if not args.no_csv:
            csv_file = args.session_dir / "summary.csv"
            if csv_file.exists():
                print(f"📊 CSV summary: {csv_file}")

        if not args.no_jsonl:
            jsonl_file = args.session_dir / "inference_results.jsonl"
            if jsonl_file.exists():
                print(f"📄 JSONL inference: {jsonl_file}")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()