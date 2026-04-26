#!/usr/bin/env python3
"""
MIA Flow to Original Format Converter

Converts MIA inference results from the flow version format to the original MIA format
so they can be used with the original MIA evaluation scripts.

Usage:
    python convert_mia_flow_to_original.py logs/mia/session_20250924_134153_299/inference_results.jsonl
    python convert_mia_flow_to_original.py input.jsonl --output output.jsonl
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import re


def extract_final_answer(flow_response: str) -> str:
    """
    Extract the final answer from flow response format.

    Flow responses contain multiple steps with the final answer in format:
    "Step N: FINAL ANSWER: [actual answer]"

    Args:
        flow_response: The flow response text

    Returns:
        Extracted final answer or original text if no pattern found
    """
    # Look for "FINAL ANSWER:" pattern
    final_answer_pattern = r"FINAL ANSWER:\s*(.*?)(?:\n|$|Confidence:|The reasoning process)"
    match = re.search(final_answer_pattern, flow_response, re.DOTALL | re.IGNORECASE)

    if match:
        answer = match.group(1).strip()
        # Clean up any trailing metadata
        answer = re.sub(r'\s*Confidence:.*$', '', answer, flags=re.DOTALL | re.IGNORECASE)
        answer = re.sub(r'\s*The reasoning process.*$', '', answer, flags=re.DOTALL | re.IGNORECASE)
        return answer.strip()

    # Fallback: Look for last step content
    step_pattern = r"Step \d+:(.*?)(?=Step \d+:|$)"
    steps = re.findall(step_pattern, flow_response, re.DOTALL)
    if steps:
        last_step = steps[-1].strip()
        # Try to extract clean answer from last step
        if "FINAL ANSWER:" in last_step.upper():
            # Already handled above
            pass
        else:
            # Return the last step content
            return last_step

    # If no pattern found, return original text
    return flow_response


def convert_flow_result_to_original(flow_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single flow result to original MIA format.

    Args:
        flow_result: Flow format result dictionary

    Returns:
        Original format result dictionary
    """
    # Extract the core information
    annotation = flow_result.get("annotation", {})
    result = flow_result.get("result", {})

    # Get the image URL from annotation
    # The original format uses image URL as question_id
    image_url = annotation.get("id", "")  # MIA uses "id" field for image URL

    # Get the instruction/prompt
    instruction = annotation.get("instruction", "")

    # Extract the generated response
    flow_response = result.get("gen", "")

    # Clean the response to get just the final answer
    cleaned_response = extract_final_answer(flow_response)

    # Return in original MIA format
    return {
        "question_id": image_url,
        "prompt": instruction,
        "text": cleaned_response
    }


def convert_file(input_path: Path, output_path: Path) -> None:
    """
    Convert a JSONL file from flow format to original format.

    Args:
        input_path: Path to input JSONL file (flow format)
        output_path: Path to output JSONL file (original format)
    """
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    converted_results = []
    errors = []

    print(f"🔄 Converting {input_path} to original MIA format...")

    # Read and convert each line
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                flow_result = json.loads(line.strip())
                original_result = convert_flow_result_to_original(flow_result)
                converted_results.append(original_result)
            except Exception as e:
                error_info = {
                    "line": line_num,
                    "error": str(e),
                    "content": line.strip()[:100] + "..." if len(line.strip()) > 100 else line.strip()
                }
                errors.append(error_info)
                print(f"⚠️  Warning: Failed to convert line {line_num}: {e}")

    # Save converted results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in converted_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Report results
    print(f"\n✅ Conversion completed!")
    print(f"   📊 Total lines processed: {line_num}")
    print(f"   ✅ Successfully converted: {len(converted_results)}")
    print(f"   ❌ Failed conversions: {len(errors)}")
    print(f"   💾 Output saved to: {output_path}")

    if errors:
        print(f"\n❌ Conversion errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   Line {error['line']}: {error['error']}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")

    # Show sample of converted data
    if converted_results:
        print(f"\n📝 Sample of converted format:")
        sample = converted_results[0]
        print(f"   question_id: {sample['question_id'][:50]}...")
        print(f"   prompt: {sample['prompt'][:80]}...")
        print(f"   text: {sample['text'][:80]}...")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert MIA flow format results to original MIA format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSONL file in flow format"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSONL file in original format (default: input_file with '_original' suffix)"
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        input_stem = args.input_file.stem
        args.output = args.input_file.parent / f"{input_stem}_original.jsonl"

    # Convert the file
    convert_file(args.input_file, args.output)


if __name__ == "__main__":
    main()