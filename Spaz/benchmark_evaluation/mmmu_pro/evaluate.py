#!/usr/bin/env python3
"""
MMMU PRO evaluation script.
Usage: python evaluate.py [--input INPUT_FILE] [--output OUTPUT_DIR]

This script evaluates model inference results against ground truth answers
and generates comprehensive evaluation reports following the official MMMU PRO format.
"""

import os
import sys
import json
import argparse
import glob
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime


class MMMUProEvaluator:
    """MMMU PRO evaluation engine."""
    
    def __init__(self, output_dir: str = "./output"):
        """Initialize evaluator.
        
        Args:
            output_dir: Directory containing inference results and for saving evaluation reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_results(self, input_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load inference results from file.
        
        Args:
            input_file: Specific file to load. If None, loads the most recent .jsonl file
            
        Returns:
            List of result dictionaries
        """
        if input_file is None:
            # Find the most recent .jsonl file in output directory
            pattern = os.path.join(self.output_dir, "mmmu_pro_*.jsonl")
            files = glob.glob(pattern)
            
            if not files:
                raise FileNotFoundError(f"No inference result files found in {self.output_dir}")
            
            # Sort by modification time, get the most recent
            input_file = max(files, key=os.path.getmtime)
            print(f"Loading results from: {input_file}")
        else:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
        
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    results.append(result)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(results)} results from {input_file}")
        return results
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.
        
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
        import re
        letter_match = re.search(r'\b([A-J])\b', answer)
        if letter_match:
            return letter_match.group(1)
        
        return answer
    
    def evaluate_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single result.
        
        Args:
            result: Single result dictionary
            
        Returns:
            Evaluation dictionary
        """
        if 'error' in result:
            return {
                'id': result['id'],
                'subject': result.get('subject', ''),
                'subset': result.get('subset', ''),
                'correct': False,
                'error': True,
                'ground_truth': '',
                'predicted': '',
                'normalized_ground_truth': '',
                'normalized_predicted': ''
            }
        
        ground_truth = self.normalize_answer(result.get('ground_truth', ''))
        predicted = self.normalize_answer(result.get('extracted_answer', ''))
        
        # Check if prediction is correct
        correct = ground_truth == predicted
        
        return {
            'id': result['id'],
            'subject': result.get('subject', ''),
            'subset': result.get('subset', ''),
            'topic_difficulty': result.get('topic_difficulty', ''),
            'correct': correct,
            'error': False,
            'ground_truth': result.get('ground_truth', ''),
            'predicted': result.get('extracted_answer', ''),
            'normalized_ground_truth': ground_truth,
            'normalized_predicted': predicted,
            'model_response': result.get('model_response', ''),
            'inference_time': result.get('inference_time', 0)
        }
    
    def calculate_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics.
        
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
        
        # Difficulty-wise accuracy (if available)
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
            'by_difficulty': difficulty_accuracies
        }
    
    def generate_report(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> str:
        """Generate evaluation report.
        
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
                'model_name': first_result.get('model_name', 'Unknown'),
                'mode': first_result.get('mode', 'Unknown'),
                'setting': first_result.get('setting', 'Unknown')
            }
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MMMU PRO Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model information
        report_lines.append("Model Information:")
        report_lines.append(f"  Model: {model_info.get('model_name', 'Unknown')}")
        report_lines.append(f"  Mode: {model_info.get('mode', 'Unknown')}")
        report_lines.append(f"  Setting: {model_info.get('setting', 'Unknown')}")
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
    
    def evaluate(self, input_file: Optional[str] = None) -> Dict[str, Any]:
        """Run complete evaluation.
        
        Args:
            input_file: Input file path. If None, uses most recent file
            
        Returns:
            Complete evaluation results
        """
        print("Starting MMMU PRO evaluation...")
        
        # Load results
        results = self.load_results(input_file)
        
        # Evaluate each result
        print("Evaluating results...")
        evaluations = [self.evaluate_single_result(result) for result in results]
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.calculate_metrics(evaluations)
        
        # Generate report
        report = self.generate_report(results, metrics)
        
        # Print report to console
        print("\n" + report)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"mmmu_pro_evaluation_report_{timestamp}.txt"
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
        # Save metrics as JSON
        metrics_filename = f"mmmu_pro_evaluation_metrics_{timestamp}.json"
        metrics_path = os.path.join(self.output_dir, metrics_filename)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {metrics_path}")
        
        # Save detailed results
        detailed_filename = f"mmmu_pro_evaluation_{timestamp}"
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
    parser = argparse.ArgumentParser(description='MMMU PRO evaluation script')
    parser.add_argument('--input', help='Input JSONL file with inference results')
    parser.add_argument('--output', default='./output', help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MMMUProEvaluator(args.output)
    
    # Run evaluation
    try:
        results = evaluator.evaluate(args.input)
        print(f"\n✓ Evaluation completed successfully!")
        print(f"Overall accuracy: {results['metrics']['overall']['accuracy']:.4f} ({results['metrics']['overall']['accuracy']*100:.2f}%)")
        return 0
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

