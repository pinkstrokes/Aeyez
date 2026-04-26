#!/usr/bin/env python3
"""
Re-evaluate inference results using OFFICIAL MMMU answer extraction logic
Source: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/evaluate.py

This script re-evaluates existing inference results (*.jsonl) using the official
answer extraction method to ensure comparability with published benchmarks.
"""

import json
import argparse
import pandas as pd
import re
from typing import List, Dict, Any


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices

    From MMMU official code:
    https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/eval_utils.py
    """
    start_chr = 'A'
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index, e.g., A, B, C, D.

    OFFICIAL MMMU answer extraction logic from:
    https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/evaluate.py
    """
    # First try: Look for "Answer:" pattern
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        # Extract the string after "Answer:"
        answer_str = response[last_answer_pos + len("Answer:"):].strip()

        # Find a unique match in the options
        matching_options = [option for option in all_choices if option in answer_str]

        # If a unique match is found, return that option
        if len(matching_options) == 1:
            return matching_options[0]

    # Clean the response
    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
    else:
        response = ""

    candidates = []

    # Try pattern: (A) (B) (C) (D)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    if len(candidates) == 0:
        # Try pattern: A B C D (with space)
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        # Try pattern: A. B. C. D.
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # If all above doesn't get candidates, check if the content is larger than 5 tokens
    # and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)

    # If we have exactly one candidate, return it
    if len(candidates) == 1:
        return candidates[0]

    # If multiple or no candidates, return empty string
    return ""


def evaluate_mmmu(input_file: str, output_file: str, dataset: str = "MMMU"):
    """
    Re-evaluate MMMU results using official answer extraction

    Args:
        input_file: Path to inference results JSONL (from run_mmmu.py or run_mmmu_pro.py)
        output_file: Path to save re-evaluated results CSV
        dataset: "MMMU" or "MMMU-Pro"
    """
    print("="*60)
    print(f"Re-evaluating {dataset} with OFFICIAL answer extraction")
    print("="*60)

    # Load inference results
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))

    print(f"Loaded {len(results)} inference results from {input_file}")

    # Re-evaluate with official extraction
    evaluations = []
    correct = 0
    total = 0
    errors = 0
    extraction_changed = 0

    for result in results:
        # Handle different formats: MMMU vs MMMU-Pro
        if 'annotation' in result:
            # MMMU format: nested structure
            annotation = result['annotation']
            model_response = result.get('result', {}).get('gen', '')
            question_id = result.get('question_id', '')

            # Extract ground truth and options from annotation
            ground_truth = str(annotation.get('answer', ''))

            # Build options list from A, B, C, D fields
            options = []
            for ch in ['A', 'B', 'C', 'D', 'E']:
                if ch in annotation:
                    val = annotation[ch]
                    # Check if value is not NaN or empty
                    if pd.notna(val) and str(val).strip():
                        options.append(str(val))

            subject = annotation.get('category', '')
            topic_difficulty = ''
            our_extracted = ''  # MMMU doesn't have extracted_answer field

        else:
            # MMMU-Pro format: flat structure
            if 'error' in result:
                errors += 1
                continue

            question_id = result.get('id', '')
            options = result.get('options', [])
            model_response = result.get('model_response', '')
            ground_truth = result.get('ground_truth', '')
            our_extracted = result.get('extracted_answer', '')
            subject = result.get('subject', '')
            topic_difficulty = result.get('topic_difficulty', '')

        total += 1

        # Parse options if string
        if isinstance(options, str):
            import ast
            try:
                options = ast.literal_eval(options)
            except:
                options = []

        # Skip if no options
        if not options:
            print(f"Warning: No options for {question_id}, skipping")
            total -= 1
            continue

        # Official extraction
        index2ans, all_choices = get_multi_choice_info(options)

        # Use official extraction logic
        official_extracted = parse_multi_choice_response(model_response, all_choices, index2ans)

        # Compare with our extraction (if available)
        if our_extracted and official_extracted != our_extracted:
            extraction_changed += 1

        # Check correctness
        ground_truth = str(ground_truth).strip().upper()
        official_extracted_upper = official_extracted.strip().upper()

        is_correct = (ground_truth == official_extracted_upper)
        if is_correct:
            correct += 1

        evaluation = {
            'id': question_id,
            'subject': subject,
            'topic_difficulty': topic_difficulty,
            'ground_truth': ground_truth,
            'extracted_answer': official_extracted_upper,
            'correct': is_correct
        }
        evaluations.append(evaluation)

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    # Save results
    df = pd.DataFrame(evaluations)
    df.to_csv(output_file, index=False)

    # Save accuracy metrics
    acc_file = output_file.replace('.csv', '_acc.json')
    with open(acc_file, 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors,
            "extraction_changed_count": extraction_changed,
            "extraction_changed_rate": extraction_changed / total if total > 0 else 0
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"OFFICIAL EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Errors: {errors}")
    if extraction_changed > 0:
        print(f"\nExtraction comparison:")
        print(f"  Changed from our method: {extraction_changed}/{total} ({extraction_changed/total*100:.1f}%)")
    print(f"\nResults saved to: {output_file}")
    print(f"Metrics saved to: {acc_file}")
    print(f"{'='*60}\n")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate MMMU/MMMU-Pro results with official answer extraction"
    )
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSONL file with inference results")
    parser.add_argument("--output", type=str, required=True,
                       help="Output CSV file for re-evaluated results")
    parser.add_argument("--dataset", type=str, default="MMMU",
                       choices=["MMMU", "MMMU-Pro"],
                       help="Dataset type")

    args = parser.parse_args()

    evaluate_mmmu(args.input, args.output, args.dataset)


if __name__ == "__main__":
    main()
