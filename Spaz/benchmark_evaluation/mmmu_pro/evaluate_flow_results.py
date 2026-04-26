#!/usr/bin/env python3
"""
MMMU PRO flow evaluation script.
Usage: python evaluate_flow_results.py [--results-dir RESULTS_DIR] [--output-dir OUTPUT_DIR]

This script evaluates flow-generated MMMU PRO results by reading individual validation_*.json files
and applying the same evaluation logic as the original evaluate.py script.
"""

import os
import sys
import json
import argparse
import glob
import requests
import time
import random
import copy
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import re


class DashScopeWrapper:
    """Wrapper for DashScope API for model-based answer extraction."""

    def __init__(self, model="qwen-flash", timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = os.environ.get('DASHSCOPE_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
        self.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'

        if not self.api_key:
            print("Warning: DASHSCOPE_API_KEY not found in environment variables")

    def generate(self, prompt: str) -> str:
        """Generate a response from the API."""
        if not self.api_key:
            return self.fail_msg

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 100,
            "temperature": 0,
            "stream": False
        }

        for i in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    resp_json = response.json()

                    # Check finish reason
                    for output in resp_json['choices']:
                        if output['finish_reason'] not in ['stop', 'function_call']:
                            print(f"DashScope finished with error: {resp_json}")
                            time.sleep(self.wait)
                            continue

                    return resp_json['choices'][0]['message']['content']
                else:
                    print(f"DashScope API error: HTTP {response.status_code}")

                time.sleep(self.wait)
            except Exception as e:
                print(f"DashScope error: {e}")
                time.sleep(self.wait)

        return self.fail_msg


class MMMUProFlowEvaluator:
    """MMMU PRO flow evaluation engine."""

    def __init__(self, output_dir: str = "./evaluation_results", use_model_extraction: bool = True):
        """Initialize evaluator.

        Args:
            output_dir: Directory for saving evaluation reports
            use_model_extraction: Whether to use model-based answer extraction as fallback
        """
        self.output_dir = output_dir
        self.use_model_extraction = use_model_extraction
        os.makedirs(output_dir, exist_ok=True)

        # Initialize model-based extractor if enabled
        self.model_extractor = None
        if self.use_model_extraction:
            try:
                self.model_extractor = DashScopeWrapper()
                print("✅ Model-based correctness evaluation enabled with qwen-flash")
                print("🎯 The model will directly judge if responses are correct vs ground truth")
            except Exception as e:
                print(f"⚠️ Failed to initialize model extractor: {e}")
                print("🔄 Will fall back to rule-based comparison only")
                self.use_model_extraction = False

    def load_flow_results(self, results_dir: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load flow-generated results from validation_*.json files.

        Args:
            results_dir: Directory containing validation_*.json files

        Returns:
            Tuple of (converted results, original flow results)
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # Find all validation_*.json files
        pattern = str(results_path / "validation_*.json")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(f"No validation_*.json files found in {results_dir}")

        # Sort files by their numeric index
        def extract_number(filename):
            match = re.search(r'validation_(\d+)\.json', filename)
            return int(match.group(1)) if match else 0

        files.sort(key=extract_number)
        print(f"Found {len(files)} validation files")

        converted_results = []
        original_flows = []
        for file_path in tqdm(files, desc="Loading flow results"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    flow_result = json.load(f)

                # Keep original flow data for model-based evaluation
                original_flows.append(flow_result)

                # Convert flow format to original evaluation format
                converted_result = self.convert_flow_result(flow_result)
                converted_results.append(converted_result)

            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

        print(f"Loaded {len(converted_results)} results from {results_dir}")
        return converted_results, original_flows

    def convert_flow_result(self, flow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flow result format to original evaluation format.

        Args:
            flow_result: Flow-generated result dictionary

        Returns:
            Converted result dictionary matching original format
        """
        # Extract basic information
        question_id = flow_result.get('question_id', '')
        model_response = flow_result.get('model_response', '')
        expected_answer = flow_result.get('expected_answer', '')
        metadata = flow_result.get('metadata', {})

        # Check if there's an error
        if 'critical_errors' in flow_result and flow_result['critical_errors']:
            return {
                'id': question_id,
                'mmmu_pro_id': metadata.get('mmmu_pro_id', ''),
                'subject': metadata.get('subject', ''),
                'subset': metadata.get('subset', ''),
                'topic_difficulty': metadata.get('topic_difficulty', ''),
                'ground_truth': expected_answer,
                'model_response': model_response,
                'extracted_answer': '',
                'error': f"Critical errors: {flow_result['critical_errors']}",
                'inference_time': flow_result.get('duration_seconds', 0),
                'model_name': 'flow_model',
                'mode': 'flow',
                'setting': 'standard'
            }

        # Extract final answer using enhanced extraction
        extracted_answer, extraction_method = self.extract_final_answer_enhanced(flow_result)

        # Convert to original format
        converted_result = {
            'id': question_id,
            'mmmu_pro_id': metadata.get('mmmu_pro_id', ''),
            'subject': metadata.get('subject', ''),
            'subset': metadata.get('subset', ''),
            'topic_difficulty': metadata.get('topic_difficulty', ''),
            'ground_truth': expected_answer,
            'model_response': model_response,
            'extracted_answer': extracted_answer,
            'extraction_method': extraction_method,
            'inference_time': flow_result.get('duration_seconds', 0),
            'model_name': 'flow_model',
            'mode': 'flow',
            'setting': 'standard'
        }

        return converted_result

    def extract_final_answer(self, model_response: str) -> str:
        """Extract the final answer from flow model response.

        Args:
            model_response: Raw model response from flow

        Returns:
            Extracted answer letter
        """
        if not model_response:
            return ""

        # Look for "FINAL ANSWER:" pattern
        response_lines = model_response.split('\n')
        for line in response_lines:
            if 'FINAL ANSWER:' in line.upper():
                # Extract text after "FINAL ANSWER:"
                parts = line.split(':', 1)
                if len(parts) > 1:
                    answer = parts[1].strip()

                    # Look for single letter answer pattern
                    letter_match = re.search(r'\b([A-J])\b', answer)
                    if letter_match:
                        return letter_match.group(1)

                    # Look for pattern like "A. Something"
                    option_match = re.search(r'\b([A-J])\.\s*', answer)
                    if option_match:
                        return option_match.group(1)

                    # If nothing found, try to extract first letter
                    if len(answer) > 0 and answer[0].upper() in 'ABCDEFGHIJ':
                        return answer[0].upper()

        # Fallback: look for any single letter in the response
        letter_match = re.search(r'\b([A-J])\b', model_response)
        if letter_match:
            return letter_match.group(1)

        return ""

    def can_infer_option(self, answer: str, choices: List[str]) -> str:
        """Rule-based extraction of answer option (adapted from MMMU)."""
        if 'Failed to obtain answer via API' in answer:
            return ""

        reject_to_answer = [
            "Sorry, I can't help with images of people yet.",
            "I can't process this file.",
            "I'm sorry, but without the image provided",
            'Cannot determine the answer'
        ]
        for err in reject_to_answer:
            if err in answer:
                return 'Z'

        def count_choice(splits, choices, prefix='', suffix=''):
            cnt = 0
            for c in choices:
                if prefix + c + suffix in splits:
                    cnt += 1
            return cnt

        answer_mod = copy.copy(answer)
        chars = '.()[],:;!*#{}'
        for c in chars:
            answer_mod = answer_mod.replace(c, ' ')

        splits = [x.strip() for x in answer_mod.split()]
        count = count_choice(splits, choices)

        if count == 1:
            for ch in choices:
                if 'A' in splits and len(splits) > 3:
                    return ""  # A might be a quantifier
                if ch in splits:
                    return ch
        elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
            return 'Z'
        return ""

    def can_infer_text(self, answer: str, choices: Dict[str, str]) -> str:
        """Extract answer by matching text content (adapted from MMMU)."""
        answer = answer.lower()
        cands = []
        for k, v in choices.items():
            if str(v).lower() in answer:
                cands.append(k)
        if len(cands) == 1:
            return cands[0]
        return ""

    def can_infer(self, answer: str, choices: Dict[str, str]) -> str:
        """Combined rule-based approach to infer answer choice."""
        answer = str(answer)
        choice_keys = list(choices.keys())
        copt = self.can_infer_option(answer, choice_keys)
        return copt if copt else self.can_infer_text(answer, choices)

    def build_option_str(self, choices: Dict[str, str]) -> str:
        """Build options string for model-based extraction."""
        s = ""
        for c, content in choices.items():
            if content:
                s += f'{c}. {content} '
        return s.strip()

    def build_extraction_prompt(self, question: str, options: str, prediction: str) -> str:
        """Build prompt for model-based answer extraction."""
        tmpl = (
            'You are an AI assistant who will help me to match '
            'an answer with several options of a single-choice question. '
            'You are provided with a question, several options, and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different from the answer, output Z. '
            'Your should output a single uppercase character in A, B, C, D, E, F, G, H, I, J (if they are valid options), and Z. \n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
            'Answer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
            'Answer: Spider\nYour output: Z\n'
            'Example 3: \n'
            'Question: {}\nOptions: {}\nAnswer: {}\nYour output: '
        )
        return tmpl.format(question, options, prediction)

    def build_evaluation_prompt(self, question: str, options: str, prediction: str, ground_truth: str) -> str:
        """Build prompt for model-based correctness evaluation."""
        tmpl = (
            'You are an AI assistant who will help me evaluate whether a model\'s answer to a multiple-choice question is correct. '
            'You are provided with a question, several options, the model\'s response, and the correct answer. '
            'Your task is to determine if the model\'s response is semantically equivalent to the correct answer, '
            'even if the wording is different. Consider the meaning and intent rather than exact text matching.\n\n'
            'Please respond with exactly "CORRECT" if the model\'s answer matches the ground truth, '
            'or "INCORRECT" if it does not match.\n\n'
            'Example 1:\n'
            'Question: What is the main object in the image?\n'
            'Options: (A) teddy bear (B) rabbit (C) cat (D) dog\n'
            'Ground Truth: A\n'
            'Model Response: The main object in the image is a cute teddy bear sitting on a chair.\n'
            'Your output: CORRECT\n\n'
            'Example 2:\n'
            'Question: What color is the car?\n'
            'Options: (A) red (B) blue (C) green (D) yellow\n'
            'Ground Truth: B\n'
            'Model Response: The car appears to be painted in a bright red color.\n'
            'Your output: INCORRECT\n\n'
            'Now evaluate:\n'
            'Question: {}\n'
            'Options: {}\n'
            'Ground Truth: {}\n'
            'Model Response: {}\n'
            'Your output: '
        )
        return tmpl.format(question, options, ground_truth, prediction)

    def evaluate_correctness_with_model(self, flow_result: Dict[str, Any], ground_truth: str) -> tuple[bool, str]:
        """Use model to directly evaluate if the response is correct."""
        if not self.use_model_extraction or not self.model_extractor:
            return False, "model_unavailable"

        model_response = flow_result.get('model_response', '')
        options = flow_result.get('options', [])
        raw_question = flow_result.get('raw_question', '')

        # Build options string
        options_str = ""
        for i, option in enumerate(options):
            if option:
                options_str += f"{option} "

        if not options_str.strip():
            return False, "no_options"

        # Build evaluation prompt
        prompt = self.build_evaluation_prompt(raw_question, options_str.strip(), model_response, ground_truth)

        print(f"🔄 Using model-based correctness evaluation...")

        retry = 3
        while retry > 0:
            try:
                model_evaluation = self.model_extractor.generate(prompt)

                if 'Failed to obtain answer via API' not in model_evaluation:
                    # Parse model response
                    evaluation_clean = model_evaluation.strip().upper()

                    if "CORRECT" in evaluation_clean:
                        print(f"✅ Model evaluation: CORRECT")
                        return True, "model_eval"
                    elif "INCORRECT" in evaluation_clean:
                        print(f"❌ Model evaluation: INCORRECT")
                        return False, "model_eval"
                    else:
                        print(f"⚠️ Model evaluation unclear: {model_evaluation}")

            except Exception as e:
                print(f"⚠️ Model evaluation error: {e}")

            retry -= 1
            time.sleep(random.random() * 2)

        print(f"❌ Model evaluation failed")
        return False, "model_eval_failed"

    def extract_final_answer_enhanced(self, flow_result: Dict[str, Any]) -> tuple[str, str]:
        """Enhanced answer extraction using model-based approach primarily."""
        model_response = flow_result.get('model_response', '')
        options = flow_result.get('options', [])
        raw_question = flow_result.get('raw_question', '')

        # Build choices dictionary from options
        choices = {}
        for i, option in enumerate(options):
            if option:
                # Extract option letter and text
                option_match = re.match(r'\(([A-J])\)\s*(.*)', option.strip())
                if option_match:
                    letter, text = option_match.groups()
                    choices[letter] = text.strip()

        if not choices:
            # Fallback to basic extraction if no structured options
            return self.extract_final_answer(model_response), "basic"

        # Step 1: Rule-based extraction (primary - following MMMU standard)
        prediction = self.extract_final_answer(model_response)
        if prediction:
            rule_result = self.can_infer(prediction, choices)
            if rule_result and rule_result != 'Z':
                print(f"✅ Rule-based extraction succeeded: {rule_result}")
                return rule_result, "rule"
            else:
                print(f"⚠️ Rule extraction returned: {rule_result} from prediction: {prediction}")

        # Step 2: Model-based extraction fallback (only if rule fails - following MMMU standard)
        if self.use_model_extraction and self.model_extractor:
            print(f"🔄 Rule extract failed. Using model-based extraction (fallback)...")

            options_str = self.build_option_str(choices)
            prompt = self.build_extraction_prompt(raw_question, options_str, model_response)

            retry = 5
            while retry > 0:
                try:
                    model_response_extraction = self.model_extractor.generate(prompt)

                    if 'Failed to obtain answer via API' not in model_response_extraction:
                        model_result = self.can_infer(model_response_extraction, choices)
                        if model_result and model_result != 'Z':
                            print(f"✅ Model-based extraction succeeded: {model_result}")
                            return model_result, "model_fallback"
                        else:
                            print(f"⚠️ Model extraction returned: {model_response_extraction}")

                except Exception as e:
                    print(f"⚠️ Model extraction error: {e}")

                retry -= 1
                time.sleep(random.random() * 2)

            print(f"❌ Model-based extraction failed...")

        # Step 3: Try rule-based extraction on full response
        rule_result = self.can_infer(model_response, choices)
        if rule_result and rule_result != 'Z':
            print(f"✅ Rule-based extraction on full response succeeded: {rule_result}")
            return rule_result, "rule_full_fallback"

        # Step 4: Basic extraction fallback
        basic_result = self.extract_final_answer(model_response)
        if basic_result and basic_result in choices:
            print(f"⚠️ Using basic extraction fallback: {basic_result}")
            return basic_result, "basic_fallback"

        # Last resort: random choice
        if choices:
            fallback = random.choice(list(choices.keys()))
            print(f"⚠️ Using random fallback: {fallback}")
            return fallback, "random"

        return "", "failed"

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison (same as original).

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer
        """
        if not isinstance(answer, str):
            answer = str(answer)

        # Convert to uppercase and strip whitespace
        answer = answer.strip().upper()

        # For single letter answers, extract the letter
        if len(answer) == 1 and answer.isalpha():
            return answer

        # Look for letter patterns
        letter_match = re.search(r'\b([A-J])\b', answer)
        if letter_match:
            return letter_match.group(1)

        return answer

    def evaluate_single_result(self, result: Dict[str, Any], flow_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate a single result using model-based correctness evaluation.

        Args:
            result: Single result dictionary
            flow_result: Original flow result for model-based evaluation

        Returns:
            Evaluation dictionary
        """
        if 'error' in result:
            return {
                'id': result['id'],
                'subject': result.get('subject', ''),
                'subset': result.get('subset', ''),
                'topic_difficulty': result.get('topic_difficulty', ''),
                'correct': False,
                'error': True,
                'ground_truth': result.get('ground_truth', ''),
                'predicted': result.get('extracted_answer', ''),
                'normalized_ground_truth': '',
                'normalized_predicted': '',
                'model_response': result.get('model_response', ''),
                'inference_time': result.get('inference_time', 0),
                'evaluation_method': 'error',
                'extraction_method': result.get('extraction_method', 'rule'),
                'extraction_success': bool(result.get('extracted_answer', ''))
            }

        ground_truth = result.get('ground_truth', '')
        predicted = result.get('extracted_answer', '')

        # Always use rule-based exact matching for correctness evaluation (following official MMMU-Pro)
        # Model-based evaluation is only used for answer extraction, not for judging correctness
        normalized_gt = self.normalize_answer(ground_truth)
        normalized_pred = self.normalize_answer(predicted)
        correct = normalized_gt == normalized_pred
        evaluation_method = 'rule_based'

        return {
            'id': result['id'],
            'subject': result.get('subject', ''),
            'subset': result.get('subset', ''),
            'topic_difficulty': result.get('topic_difficulty', ''),
            'correct': correct,
            'error': False,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'normalized_ground_truth': self.normalize_answer(ground_truth),
            'normalized_predicted': self.normalize_answer(predicted),
            'model_response': result.get('model_response', ''),
            'inference_time': result.get('inference_time', 0),
            'evaluation_method': evaluation_method,
            'extraction_method': result.get('extraction_method', 'rule'),
            'extraction_success': bool(predicted)
        }

    def calculate_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics (same as original).

        Args:
            evaluations: List of evaluation dictionaries

        Returns:
            Metrics dictionary
        """
        total_samples = len(evaluations)
        correct_samples = sum(1 for e in evaluations if e['correct'])
        error_samples = sum(1 for e in evaluations if e['error'])
        successful_samples = total_samples - error_samples

        # Overall accuracy
        overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0
        successful_accuracy = correct_samples / successful_samples if successful_samples > 0 else 0

        # Subject-wise accuracy
        subject_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0})
        for eval_result in evaluations:
            subject = eval_result['subject']
            subject_stats[subject]['total'] += 1
            if eval_result['error']:
                subject_stats[subject]['errors'] += 1
            elif eval_result['correct']:
                subject_stats[subject]['correct'] += 1

        subject_accuracies = {}
        for subject, stats in subject_stats.items():
            if stats['total'] > 0:
                subject_accuracies[subject] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'successful_accuracy': stats['correct'] / (stats['total'] - stats['errors']) if (stats['total'] - stats['errors']) > 0 else 0,
                    'total_samples': stats['total'],
                    'correct_samples': stats['correct'],
                    'error_samples': stats['errors']
                }

        # Subset-wise accuracy
        subset_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0})
        for eval_result in evaluations:
            subset = eval_result['subset']
            subset_stats[subset]['total'] += 1
            if eval_result['error']:
                subset_stats[subset]['errors'] += 1
            elif eval_result['correct']:
                subset_stats[subset]['correct'] += 1

        subset_accuracies = {}
        for subset, stats in subset_stats.items():
            if stats['total'] > 0:
                subset_accuracies[subset] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'successful_accuracy': stats['correct'] / (stats['total'] - stats['errors']) if (stats['total'] - stats['errors']) > 0 else 0,
                    'total_samples': stats['total'],
                    'correct_samples': stats['correct'],
                    'error_samples': stats['errors']
                }

        # Difficulty-wise accuracy
        difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0})
        for eval_result in evaluations:
            difficulty = eval_result.get('topic_difficulty', 'Unknown')
            if difficulty:
                difficulty_stats[difficulty]['total'] += 1
                if eval_result['error']:
                    difficulty_stats[difficulty]['errors'] += 1
                elif eval_result['correct']:
                    difficulty_stats[difficulty]['correct'] += 1

        difficulty_accuracies = {}
        for difficulty, stats in difficulty_stats.items():
            if stats['total'] > 0:
                difficulty_accuracies[difficulty] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'successful_accuracy': stats['correct'] / (stats['total'] - stats['errors']) if (stats['total'] - stats['errors']) > 0 else 0,
                    'total_samples': stats['total'],
                    'correct_samples': stats['correct'],
                    'error_samples': stats['errors']
                }

        # Extraction method statistics
        extraction_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for eval_result in evaluations:
            if not eval_result['error']:
                method = eval_result.get('extraction_method', 'unknown')
                extraction_stats[method]['total'] += 1
                if eval_result['correct']:
                    extraction_stats[method]['correct'] += 1

        extraction_accuracies = {}
        for method, stats in extraction_stats.items():
            if stats['total'] > 0:
                extraction_accuracies[method] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'total_samples': stats['total'],
                    'correct_samples': stats['correct']
                }

        # Evaluation method statistics
        evaluation_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for eval_result in evaluations:
            if not eval_result['error']:
                method = eval_result.get('evaluation_method', 'unknown')
                evaluation_stats[method]['total'] += 1
                if eval_result['correct']:
                    evaluation_stats[method]['correct'] += 1

        evaluation_method_accuracies = {}
        for method, stats in evaluation_stats.items():
            if stats['total'] > 0:
                evaluation_method_accuracies[method] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'total_samples': stats['total'],
                    'correct_samples': stats['correct']
                }

        return {
            'overall': {
                'accuracy': overall_accuracy,
                'successful_accuracy': successful_accuracy,
                'total_samples': total_samples,
                'correct_samples': correct_samples,
                'error_samples': error_samples,
                'successful_samples': successful_samples
            },
            'by_subject': subject_accuracies,
            'by_subset': subset_accuracies,
            'by_difficulty': difficulty_accuracies,
            'by_extraction_method': extraction_accuracies,
            'by_evaluation_method': evaluation_method_accuracies
        }

    def generate_report(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> str:
        """Generate evaluation report (same format as original).

        Args:
            results: List of result dictionaries
            metrics: Calculated metrics

        Returns:
            Formatted report string
        """
        # Extract metadata from first result
        model_info = {}
        if results:
            first_result = results[0]
            model_info = {
                'model_name': first_result.get('model_name', 'flow_model'),
                'mode': first_result.get('mode', 'flow'),
                'setting': first_result.get('setting', 'standard')
            }

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MMMU PRO Flow Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Model information
        report_lines.append("Model Information:")
        report_lines.append(f"  Model: {model_info.get('model_name', 'flow_model')}")
        report_lines.append(f"  Mode: {model_info.get('mode', 'flow')}")
        report_lines.append(f"  Setting: {model_info.get('setting', 'standard')}")
        report_lines.append("")

        # Overall performance
        overall = metrics['overall']
        report_lines.append("Overall Performance:")
        report_lines.append(f"  Total Samples: {overall['total_samples']}")
        report_lines.append(f"  Correct: {overall['correct_samples']}")
        report_lines.append(f"  Errors: {overall['error_samples']}")
        report_lines.append(f"  Overall Accuracy: {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
        report_lines.append(f"  Successful Accuracy: {overall['successful_accuracy']:.4f} ({overall['successful_accuracy']*100:.2f}%)")
        report_lines.append("")

        # Performance by subset
        if metrics['by_subset']:
            report_lines.append("Performance by Subset:")
            for subset, stats in sorted(metrics['by_subset'].items()):
                report_lines.append(f"  {subset}:")
                report_lines.append(f"    Samples: {stats['total_samples']} | Correct: {stats['correct_samples']} | Errors: {stats['error_samples']}")
                report_lines.append(f"    Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
                report_lines.append(f"    Successful Accuracy: {stats['successful_accuracy']:.4f} ({stats['successful_accuracy']*100:.2f}%)")
            report_lines.append("")

        # Performance by subject (top 10)
        if metrics['by_subject']:
            report_lines.append("Performance by Subject (Top 10):")
            subject_items = sorted(metrics['by_subject'].items(), key=lambda x: x[1]['accuracy'], reverse=True)[:10]
            for subject, stats in subject_items:
                report_lines.append(f"  {subject}:")
                report_lines.append(f"    Samples: {stats['total_samples']} | Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
            report_lines.append("")

        # Performance by difficulty
        if metrics['by_difficulty']:
            report_lines.append("Performance by Difficulty:")
            for difficulty, stats in sorted(metrics['by_difficulty'].items()):
                if difficulty and difficulty != 'Unknown':
                    report_lines.append(f"  {difficulty}:")
                    report_lines.append(f"    Samples: {stats['total_samples']} | Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
            report_lines.append("")

        # Performance by extraction method
        if metrics['by_extraction_method']:
            report_lines.append("Performance by Answer Extraction Method:")
            for method, stats in sorted(metrics['by_extraction_method'].items(), key=lambda x: x[1]['total_samples'], reverse=True):
                report_lines.append(f"  {method}:")
                report_lines.append(f"    Samples: {stats['total_samples']} | Correct: {stats['correct_samples']} | Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
            report_lines.append("")

        # Performance by evaluation method
        if metrics['by_evaluation_method']:
            report_lines.append("Performance by Correctness Evaluation Method:")
            for method, stats in sorted(metrics['by_evaluation_method'].items(), key=lambda x: x[1]['total_samples'], reverse=True):
                report_lines.append(f"  {method}:")
                report_lines.append(f"    Samples: {stats['total_samples']} | Correct: {stats['correct_samples']} | Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
            report_lines.append("")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def save_detailed_results(self, evaluations: List[Dict[str, Any]], filename_prefix: str):
        """Save detailed evaluation results to CSV.

        Args:
            evaluations: List of evaluation dictionaries
            filename_prefix: Prefix for output filename
        """
        # Convert to DataFrame
        df = pd.DataFrame(evaluations)

        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{filename_prefix}_detailed_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Detailed results saved to: {csv_path}")

    def evaluate(self, results_dir: str) -> Dict[str, Any]:
        """Run complete evaluation on flow results.

        Args:
            results_dir: Directory containing validation_*.json files

        Returns:
            Complete evaluation results
        """
        print("Starting MMMU PRO flow evaluation...")

        # Load flow results
        results, original_flows = self.load_flow_results(results_dir)

        # Evaluate each result
        print("Evaluating results...")
        evaluations = []
        for i, (converted_result, original_flow) in enumerate(tqdm(zip(results, original_flows), desc="Processing", total=len(results))):
            # Evaluate with both converted result and original flow data
            evaluation = self.evaluate_single_result(converted_result, original_flow)
            evaluations.append(evaluation)

        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.calculate_metrics(evaluations)

        # Generate report
        report = self.generate_report(results, metrics)

        # Print report to console
        print("\n" + report)

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"mmmu_pro_flow_evaluation_report_{timestamp}.txt"
        report_path = os.path.join(self.output_dir, report_filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        # Save metrics as JSON
        metrics_filename = f"mmmu_pro_flow_evaluation_metrics_{timestamp}.json"
        metrics_path = os.path.join(self.output_dir, metrics_filename)

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {metrics_path}")

        # Save detailed results
        detailed_filename = f"mmmu_pro_flow_evaluation_{timestamp}"
        self.save_detailed_results(evaluations, detailed_filename)

        return {
            'metrics': metrics,
            'evaluations': evaluations,
            'report': report,
            'report_file': report_path,
            'metrics_file': metrics_path
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='MMMU PRO flow evaluation script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with model-based correctness evaluation (default - uses qwen-flash to judge correctness)
  python evaluate_flow_results.py --results-dir logs/mmmu_pro/standard10 --output-dir evaluation_output

  # Evaluate with only rule-based comparison (faster, no API calls)
  python evaluate_flow_results.py --results-dir logs/mmmu_pro/standard10 --no-model-extraction

  # Evaluate with custom output directory
  python evaluate_flow_results.py --results-dir logs/mmmu_pro/standard10 --output-dir my_eval_results
        """
    )

    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing validation_*.json files from flow inference')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--use-model-extraction', action='store_true', default=True,
                       help='Use model-based correctness evaluation (default: True)')
    parser.add_argument('--no-model-extraction', action='store_true',
                       help='Disable model-based evaluation (use rule-based comparison only)')

    args = parser.parse_args()

    # Handle model extraction flags
    if args.no_model_extraction:
        args.use_model_extraction = False

    print("🚀 Starting MMMU PRO Flow Results Evaluation")
    print(f"📁 Results directory: {args.results_dir}")
    print(f"📁 Output directory: {args.output_dir}")

    # Initialize evaluator
    evaluator = MMMUProFlowEvaluator(args.output_dir, args.use_model_extraction)

    # Run evaluation
    try:
        results = evaluator.evaluate(args.results_dir)
        print(f"\n✅ Evaluation completed successfully!")
        print(f"🎯 Overall accuracy: {results['metrics']['overall']['accuracy']:.4f} ({results['metrics']['overall']['accuracy']*100:.2f}%)")
        print(f"📁 Results saved to: {args.output_dir}")
        return 0
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())