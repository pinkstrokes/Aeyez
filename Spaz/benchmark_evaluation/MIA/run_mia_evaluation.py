#!/usr/bin/env python3
"""
MIA-Bench Evaluation Runner

This script evaluates MIA-Bench inference results using the official evaluation methodology.
It integrates with the official evaluation notebook code while providing a command-line interface.

Usage:
    python run_mia_evaluation.py --results outputs/qwen3b_results.jsonl
    python run_mia_evaluation.py --results outputs/gpt4o_results.jsonl --output-dir evaluation_results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import traceback
from tqdm import tqdm

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

# Try to import OpenAI for evaluation (following official code)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  OpenAI package not available. GPT-4o evaluation will not work.")
    OPENAI_AVAILABLE = False


def load_mia_dataset(data_path: str = "data/instruction_benchmark_all.json") -> List[Dict[str, Any]]:
    """Load MIA-Bench dataset for evaluation."""
    full_path = current_dir / data_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"MIA dataset not found at {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Loaded MIA dataset: {len(data)} samples")
    return data


def load_inference_results(results_path: str) -> List[Dict[str, Any]]:
    """Load inference results from JSONL file."""
    results = []
    
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    
    print(f"📊 Loaded inference results: {len(results)} samples")
    return results


def generate_prompt(sample_data: Dict[str, Any], response: str) -> str:
    """
    Generate evaluation prompt following the official MIA evaluation methodology.
    
    This function exactly replicates the generate_prompt function from the official evaluation notebook.
    """
    instruction = sample_data['instruction']
    weight = sample_data['component_weight'] * 1  # Convert to list if needed
    sample_data['num_of_component'] = len(sample_data['components'])
    
    # Convert weights to strings
    for i in range(len(weight)):
        weight[i] = str(weight[i])
    
    # Generate components description based on number of components (exact official format)
    if sample_data['num_of_component'] == 1:
        components = '''The first component is:' ''' + sample_data['components'][0] + "'"  
        score = '''The first component is worth ''' + weight[0] + ' scores.'
    elif sample_data['num_of_component'] == 2:
        components = '''The first component is:' ''' + sample_data['components'][0] + '''', and the second component is:' ''' + sample_data['components'][1] + "'" 
        score = '''The first and second component is each worth ''' + weight[0] + ' and ' + weight[1]+ ' scores.'
    elif sample_data['num_of_component'] == 3:
        components = '''The first component is:' ''' + sample_data['components'][0] + '''', and the second component is:' ''' + sample_data['components'][1] + '''', and the third component is:' ''' + sample_data['components'][2] + "'" 
        score = '''The first second, and third component is each worth ''' + weight[0] + ', ' + weight[1]+ ' and ' + weight[2] + ' scores.'
    elif sample_data['num_of_component'] == 4:
        components = '''The first component is:' ''' + sample_data['components'][0] + '''', and the second component is:' ''' + sample_data['components'][1] + '''', and the third component is:' ''' + sample_data['components'][2] +  '''', and the fourth component is:' ''' + sample_data['components'][3] + "'" 
        score = '''The first second, third, and fourth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ' and ' + weight[3] + ' scores.'
    elif sample_data['num_of_component'] == 5:
        components = '''The first component is:' ''' + sample_data['components'][0] + '''', and the second component is:' ''' + sample_data['components'][1] + '''', and the third component is:' ''' + sample_data['components'][2] +  '''', and the fourth component is:' ''' + sample_data['components'][3] +  '''', and the fifth component is:' ''' + sample_data['components'][4] + "'" 
        score = '''The first second, third, fourth and fifth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ', ' + weight[3] + ' and ' + weight[4] + ' scores.'      
    else:
        raise ValueError(f"Unsupported number of components: {sample_data['num_of_component']}")
    
    # Return exact official format
    return '''Here is an instruction for a multimodal LLM: ' ''' + instruction + ''' You need to grade if the response from the model follows each component of the instruction. ''' + components + ''' The response is:' '''  + response +  ''''' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. ''' + score + ' List scores of each component, and the total score in one sentence in this format: score of component 1: x/2, score of component 2: y/8, total score: z/10. Then explain your reasons.'


def evaluate_with_gpt4o(dataset: List[Dict[str, Any]], 
                       results: List[Dict[str, Any]], 
                       api_key: str) -> List[Dict[str, Any]]:
    """
    Evaluate results using GPT-4o as judge (following official methodology).
    
    Args:
        dataset: Original MIA dataset
        results: Inference results to evaluate
        api_key: OpenAI API key
        
    Returns:
        List of evaluation results with scores
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package required for GPT-4o evaluation")
    
    client = OpenAI(api_key=api_key)
    
    # Create mapping from image URL to dataset sample
    dataset_map = {sample['image']: sample for sample in dataset}
    
    evaluation_results = []
    
    print(f"🔍 Evaluating {len(results)} results with GPT-4o...")
    
    # Use tqdm for progress bar
    for i, result in enumerate(tqdm(results, desc="Evaluating", unit="sample")):
        try:
            # Find corresponding dataset sample
            # The image URL is in annotation['id'] field
            annotation = result.get('annotation', {})
            image_url = annotation.get('id', '')

            if not image_url or image_url not in dataset_map:
                print(f"⚠️  Warning: No dataset sample found for question_id {result.get('question_id')} (image: {image_url})")
                continue
            
            dataset_sample = dataset_map[image_url]

            # Extract model response from result structure
            # Flow mode format: result['result']['gen']
            # API mode format: result['text']
            if 'result' in result and 'gen' in result['result']:
                response_text = result['result']['gen']
            elif 'text' in result:
                response_text = result['text']
            else:
                print(f"⚠️  Warning: No response text found for question_id {result.get('question_id')}")
                continue

            # Skip error responses
            if not response_text or response_text == "error":
                print(f"⏭️  Skipping error response for sample {i}")
                continue
            
            # Generate evaluation prompt
            eval_prompt = generate_prompt(dataset_sample, response_text)
            
            # Call GPT-4o for evaluation (following official format with image)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": eval_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.0  # Deterministic evaluation
            )
            
            evaluation_text = completion.choices[0].message.content
            
            # Store evaluation result
            eval_result = {
                "sample_idx": i,
                "image_url": image_url,
                "instruction": dataset_sample['instruction'],
                "model_response": response_text,
                "evaluation_prompt": eval_prompt,
                "evaluation_response": evaluation_text,
                "components": dataset_sample['components'],
                "component_weights": dataset_sample['component_weight'],
                "component_types": dataset_sample['component_type']
            }
            
            evaluation_results.append(eval_result)
            
            if (i + 1) % 10 == 0:
                print(f"📊 Evaluated {i + 1}/{len(results)} samples")
                
        except Exception as e:
            print(f"❌ Error evaluating sample {i}: {e}")
            continue
    
    print(f"✅ Evaluation completed: {len(evaluation_results)} samples evaluated")
    return evaluation_results


def parse_scores_from_evaluation(evaluation_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Parse scores from GPT-4o evaluation responses.
    
    This attempts to extract numerical scores from the evaluation text.
    Note: This is a simplified version - the official code may have more sophisticated parsing.
    """
    parsed_results = []
    
    for result in evaluation_results:
        eval_text = result['evaluation_response']
        
        try:
            # Try to extract total score (looking for pattern like "total score: x/10")
            import re
            
            # Look for total score pattern
            total_match = re.search(r'total score:\s*(\d+(?:\.\d+)?)/10', eval_text, re.IGNORECASE)
            total_score = float(total_match.group(1)) if total_match else None
            
            # Extract component scores (simplified - may need refinement)
            component_scores = []
            component_matches = re.findall(r'component \d+:\s*(\d+(?:\.\d+)?)/\d+', eval_text, re.IGNORECASE)
            for match in component_matches:
                component_scores.append(float(match))
            
            parsed_result = {
                "sample_idx": result["sample_idx"],
                "image_url": result["image_url"],
                "instruction": result["instruction"],
                "model_response": result["model_response"],
                "total_score": total_score,
                "total_score_normalized": total_score / 10.0 if total_score is not None else None,
                "component_scores": component_scores,
                "num_components": len(result["components"]),
                "evaluation_text": eval_text
            }
            
            parsed_results.append(parsed_result)
            
        except Exception as e:
            print(f"⚠️  Error parsing scores for sample {result['sample_idx']}: {e}")
            # Add with null scores
            parsed_result = {
                "sample_idx": result["sample_idx"],
                "image_url": result["image_url"],
                "instruction": result["instruction"],
                "model_response": result["model_response"],
                "total_score": None,
                "total_score_normalized": None,
                "component_scores": [],
                "num_components": len(result["components"]),
                "evaluation_text": eval_text
            }
            parsed_results.append(parsed_result)
    
    return pd.DataFrame(parsed_results)


def create_official_format_dataframe(dataset: List[Dict[str, Any]], 
                                   results: List[Dict[str, Any]], 
                                   evaluation_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create DataFrame in official MIA-Bench format.
    
    This matches the format used in the official evaluation.ipynb where:
    1. Start with the original dataset DataFrame
    2. Add a new column with the model's evaluation scores
    """
    # Create DataFrame from original dataset
    df = pd.DataFrame(dataset)
    
    # Create mapping from image URL to evaluation result
    eval_map = {}
    for eval_result in evaluation_results:
        image_url = eval_result['image_url']
        eval_response = eval_result['evaluation_response']
        eval_map[image_url] = eval_response
    
    # Create mapping from image URL to model response
    response_map = {}
    for result in results:
        image_url = result['question_id']  # question_id contains the image URL
        response_text = result['text']
        response_map[image_url] = response_text
    
    # Add model response column (for reference)
    model_name = "model_response"
    df[model_name] = df['image'].map(response_map)
    
    # Add evaluation score column (this is the key official format)
    score_column = "score_raw"  # Following official naming convention
    df[score_column] = df['image'].map(eval_map)
    
    return df


def generate_evaluation_report(df: pd.DataFrame, model_name: str) -> str:
    """Generate evaluation report with key metrics."""
    
    # Filter out samples with missing scores
    valid_df = df[df['total_score'].notna()]
    
    if len(valid_df) == 0:
        return "❌ No valid scores found in evaluation results."
    
    # Calculate metrics
    total_samples = len(df)
    valid_samples = len(valid_df)
    mean_score = valid_df['total_score_normalized'].mean()
    std_score = valid_df['total_score_normalized'].std()
    
    # Score distribution
    score_bins = pd.cut(valid_df['total_score_normalized'], 
                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                       labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    score_distribution = score_bins.value_counts().sort_index()
    
    report = f"""
# MIA-Bench Evaluation Report

## Model: {model_name}
## Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Overall Results
- **Total Samples**: {total_samples}
- **Successfully Evaluated**: {valid_samples}
- **Success Rate**: {valid_samples/total_samples*100:.1f}%

### Performance Metrics
- **Mean Score**: {mean_score:.3f} ({mean_score*100:.1f}%)
- **Standard Deviation**: {std_score:.3f}
- **Median Score**: {valid_df['total_score_normalized'].median():.3f}
- **Min Score**: {valid_df['total_score_normalized'].min():.3f}
- **Max Score**: {valid_df['total_score_normalized'].max():.3f}

### Score Distribution
"""
    
    for score_range, count in score_distribution.items():
        percentage = count / valid_samples * 100
        report += f"- **{score_range}**: {count} samples ({percentage:.1f}%)\n"
    
    return report


def run_evaluation(args):
    """Run MIA-Bench evaluation following official format."""
    print(f"🚀 Starting MIA-Bench evaluation")
    print(f"   Results file: {args.results}")
    print(f"   Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset and results
    dataset = load_mia_dataset(args.data_path)
    results = load_inference_results(args.results)
    
    # Check API key for GPT-4o evaluation
    api_key = os.environ.get('OPENAI_API_KEY') or args.openai_api_key
    if not api_key:
        print("❌ OpenAI API key required for evaluation. Set OPENAI_API_KEY environment variable or use --openai-api-key")
        return
    
    # Run evaluation
    try:
        evaluation_results = evaluate_with_gpt4o(dataset, results, api_key)
        
        if not evaluation_results:
            print("❌ No evaluation results generated")
            return
        
        # Create DataFrame following official format
        df_official = create_official_format_dataframe(dataset, results, evaluation_results)
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(args.results).stem
        
        # Save in official format (CSV with evaluation column added)
        official_csv_path = output_dir / f"mia_official_format_{model_name}_{timestamp}.csv"
        df_official.to_csv(official_csv_path, index=False)
        print(f"📊 Official format CSV saved to: {official_csv_path}")
        
        # Parse scores (for our analysis)
        df_parsed = parse_scores_from_evaluation(evaluation_results)
        
        # Save detailed results
        detailed_results_path = output_dir / f"mia_evaluation_detailed_{model_name}_{timestamp}.csv"
        df_parsed.to_csv(detailed_results_path, index=False)
        print(f"📊 Detailed results saved to: {detailed_results_path}")
        
        # Save raw evaluation data
        raw_results_path = output_dir / f"mia_evaluation_raw_{model_name}_{timestamp}.json"
        with open(raw_results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"📋 Raw evaluation data saved to: {raw_results_path}")
        
        # Generate and save report
        report = generate_evaluation_report(df_parsed, model_name)
        report_path = output_dir / f"mia_evaluation_report_{model_name}_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Evaluation report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print(report)
        print("="*50)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        traceback.print_exc()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate MIA-Bench inference results using official methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Qwen2.5-VL results
  python run_mia_evaluation.py --results outputs/qwen3b_results.jsonl
  
  # Evaluate with custom output directory
  python run_mia_evaluation.py --results outputs/gpt4o_results.jsonl --output-dir custom_eval
  
  # Specify OpenAI API key directly
  python run_mia_evaluation.py --results outputs/model_results.jsonl --openai-api-key sk-...
        """
    )
    
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to inference results JSONL file"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/instruction_benchmark_all.json",
        help="Path to MIA dataset JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key (can also set OPENAI_API_KEY environment variable)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.results).exists():
        print(f"❌ Results file not found: {args.results}")
        return
    
    # Run evaluation (Python 3.6 compatibility)
    try:
        # Python 3.7+ has asyncio.run, 3.6 needs manual event loop
        try:
            import asyncio
            # For async functions, use asyncio.run if available
            run_evaluation(args)
        except Exception as async_error:
            # If there are async issues, handle them
            run_evaluation(args)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
