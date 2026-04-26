#!/usr/bin/env python3
"""
Evaluate MIA-Bench results using GPT-4o as judge.
Based on official MIA-Bench evaluation: https://github.com/apple/ml-mia-bench

Key improvements:
1. Use direct image URLs (consistent with official code)
2. Simplified prompt format to avoid GPT-4o content policy issues
3. Robust error handling and retry logic
4. Frequent progress saving
"""

import json
import os
import time
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def call_gpt4o_judge(image_url, benchmark_item, model_response, max_retries=5):
    """
    Call GPT-4o API to evaluate the response using direct image URL.
    Uses the same format as official MIA-Bench code.
    """
    instruction = benchmark_item['instruction']
    components = benchmark_item['components']
    component_weight = benchmark_item['component_weight']
    num_components = len(components)
    
    # Convert weights to strings (like official code)
    weight = [str(w) for w in component_weight]
    
    # Build component description (same as official)
    if num_components == 1:
        components_desc = f"The first component is: '{components[0]}' "
        score_desc = f"The first component is worth {weight[0]} scores."
    elif num_components == 2:
        components_desc = f"The first component is: '{components[0]}', and the second component is: '{components[1]}' "
        score_desc = f"The first and second component is each worth {weight[0]} and {weight[1]} scores."
    elif num_components == 3:
        components_desc = f"The first component is: '{components[0]}', the second component is: '{components[1]}', and the third component is: '{components[2]}' "
        score_desc = f"The first, second and third component is each worth {weight[0]}, {weight[1]} and {weight[2]} scores."
    elif num_components == 4:
        components_desc = f"The first component is: '{components[0]}', the second component is: '{components[1]}', the third component is: '{components[2]}', and the fourth component is: '{components[3]}' "
        score_desc = f"The first, second, third and fourth component is each worth {weight[0]}, {weight[1]}, {weight[2]} and {weight[3]} scores."
    elif num_components == 5:
        components_desc = f"The first component is: '{components[0]}', the second component is: '{components[1]}', the third component is: '{components[2]}', the fourth component is: '{components[3]}', and the fifth component is: '{components[4]}' "
        score_desc = f"The first, second, third, fourth and fifth component is each worth {weight[0]}, {weight[1]}, {weight[2]}, {weight[3]} and {weight[4]} scores."
    else:
        raise ValueError(f"Unsupported number of components: {num_components}")
    
    # Build prompt exactly like official code
    eval_prompt = f"Here is an instruction for a multimodal LLM: '{instruction}' You need to grade if the response from the model follows each component of the instruction. {components_desc} The response is: '{model_response}' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. {score_desc} List scores of each component, and the total score in one sentence in this format: score of component 1: x/{weight[0]}, score of component 2: y/{weight[1]}, total score: z/10. Then explain your reasons."
    
    for attempt in range(max_retries):
        try:
            # Use exact same format as official code
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": eval_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠ Error calling GPT-4o (attempt {attempt+1}/{max_retries}): {error_msg}")
            
            # Check for specific errors
            if "rate_limit" in error_msg.lower():
                wait_time = min(2 ** attempt * 5, 60)  # Longer wait for rate limits
                print(f"  Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "invalid" in error_msg.lower() and "api" in error_msg.lower():
                print("  ✗ API key appears to be invalid!")
                return None
            else:
                # General error, shorter wait
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None
    return None

def process_raw_score(component_types, raw_score):
    """
    Parse GPT-4o's response to extract scores.
    Expected format: "score of component 1: x/y, score of component 2: a/b, total score: z/10"
    Improved parsing to handle various formats.
    """
    if not raw_score:
        return None

    # Check for refusal messages
    if any(phrase in raw_score.lower() for phrase in ["i'm sorry", "i cannot", "i can't"]):
        print(f"\n  ⚠ GPT-4o refused to answer: {raw_score[:100]}...")
        return None

    try:
        # Method 1: Official parsing approach
        first_sentence = raw_score.split('.')[0].split(',')
        scores = {}
        total_score = None

        for i, part in enumerate(first_sentence):
            part = part.strip().lower()
            
            # Look for "score of component X: Y/Z"
            if 'score of component' in part and ':' in part:
                try:
                    score_part = part.split(':')[1].strip()
                    if '/' in score_part:
                        numerator = score_part.split('/')[0].strip()
                        numerator = ''.join(c for c in numerator if c.isdigit() or c == '.')
                        if numerator:
                            comp_idx = len([k for k in scores.keys() if k.startswith('component_')])
                            scores[f'component_{comp_idx+1}'] = float(numerator)
                except:
                    continue
            
            # Look for "total score: X/10"
            if 'total score' in part and ':' in part:
                try:
                    score_part = part.split(':')[1].strip()
                    if '/' in score_part:
                        numerator = score_part.split('/')[0].strip()
                        numerator = ''.join(c for c in numerator if c.isdigit() or c == '.')
                        if numerator:
                            total_score = float(numerator)
                except:
                    continue

        if total_score is not None:
            scores['total_score'] = total_score
            return scores
        
        # Method 2: Fallback - look for patterns more broadly
        import re
        total_match = re.search(r'total score[:\s]+(\d+(?:\.\d+)?)\s*/\s*10', raw_score, re.IGNORECASE)
        if total_match:
            total_score = float(total_match.group(1))
            scores['total_score'] = total_score
            return scores

    except Exception as e:
        print(f"\n  ⚠ Error parsing score: {e}")
        print(f"    Raw score: {raw_score[:200]}")

    return None

def main():
    print("="*60)
    print("MIA-Bench Evaluation with GPT-4o")
    print("="*60)
    
    # Load benchmark data
    benchmark_path = "/u/hli36/MPU-RL-clean/src/multi-agent/benchmark_evaluation/MIA/data/instruction_benchmark_all.json"
    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)

    # Create lookup by image URL
    benchmark_lookup = {item['image']: item for item in benchmark_data}

    # Load inference results
    results_path = "/u/hli36/MPU-RL-clean/src/multi-agent/benchmark_evaluation/llava/outputs/mia/inference_results.jsonl"
    inference_results = []
    with open(results_path, 'r') as f:
        for line in f:
            inference_results.append(json.loads(line))

    print(f"\nLoaded {len(inference_results)} inference results")
    print(f"Loaded {len(benchmark_data)} benchmark items")

    # Filter successful results (no errors)
    successful_results = [r for r in inference_results if 'text' in r and r['text'] and 'error' not in r]
    print(f"Found {len(successful_results)} successful results to evaluate\n")

    # Check if we have previous progress
    output_path = "/u/hli36/MPU-RL-clean/src/multi-agent/benchmark_evaluation/llava/outputs/mia/eval_results_full.jsonl"
    already_evaluated = set()
    if os.path.exists(output_path):
        print(f"Found existing results, resuming...")
        with open(output_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                already_evaluated.add(data['url'])
        print(f"Already evaluated: {len(already_evaluated)} samples\n")

    # Evaluate each result
    evaluated_results = []
    total_scores = []
    image_url_failures = 0
    gpt4o_failures = 0
    parse_failures = 0

    for result in tqdm(successful_results, desc="Evaluating with GPT-4o"):
        image_url = result['url']
        
        # Skip if already evaluated
        if image_url in already_evaluated:
            continue
        
        model_response = result['text']

        # Get benchmark item
        if image_url not in benchmark_lookup:
            print(f"\n⚠ Warning: Image URL not found in benchmark: {image_url}")
            image_url_failures += 1
            continue

        benchmark_item = benchmark_lookup[image_url]

        # Call GPT-4o with direct URL and cleaner prompt format
        raw_score = call_gpt4o_judge(image_url, benchmark_item, model_response)

        if not raw_score:
            gpt4o_failures += 1
            eval_result = {
                'url': image_url,
                'instruction': benchmark_item['instruction'],
                'type': benchmark_item['type'],
                'components': benchmark_item['components'],
                'component_weight': benchmark_item['component_weight'],
                'component_type': benchmark_item['component_type'],
                'model_response': model_response,
                'gpt4o_raw_response': None,
                'parsed_scores': None,
                'error': 'GPT-4o call failed'
            }
            evaluated_results.append(eval_result)
            continue

        # Parse score
        parsed_scores = process_raw_score(benchmark_item['component_type'], raw_score)

        eval_result = {
            'url': image_url,
            'instruction': benchmark_item['instruction'],
            'type': benchmark_item['type'],
            'components': benchmark_item['components'],
            'component_weight': benchmark_item['component_weight'],
            'component_type': benchmark_item['component_type'],
            'model_response': model_response,
            'gpt4o_raw_response': raw_score,
            'parsed_scores': parsed_scores
        }

        if parsed_scores and 'total_score' in parsed_scores:
            total_scores.append(parsed_scores['total_score'])
        else:
            parse_failures += 1

        evaluated_results.append(eval_result)

        # Save progress every 10 samples
        if len(evaluated_results) % 10 == 0:
            with open(output_path, 'a') as f:
                for r in evaluated_results:
                    f.write(json.dumps(r) + '\n')
            evaluated_results = []  # Clear to avoid duplicates
            print(f"\n✓ Saved progress: {len(already_evaluated) + len(evaluated_results)} samples")

    # Save final results
    if evaluated_results:
        with open(output_path, 'a') as f:
            for r in evaluated_results:
                f.write(json.dumps(r) + '\n')

    # Calculate statistics
    if total_scores:
        avg_score = sum(total_scores) / len(total_scores)
        print(f"\n{'='*60}")
        print(f"MIA-Bench Evaluation Results (GPT-4o Judge)")
        print(f"{'='*60}")
        print(f"Total processed: {len(successful_results)}")
        print(f"Successfully scored: {len(total_scores)}")
        print(f"Image URL failures: {image_url_failures}")
        print(f"GPT-4o call failures: {gpt4o_failures}")
        print(f"Parse failures: {parse_failures}")
        print(f"Average score: {avg_score:.2f}/10 ({avg_score*10:.1f}%)")
        print(f"Max score: {max(total_scores):.2f}/10")
        print(f"Min score: {min(total_scores):.2f}/10")

        # Reload all results for breakdown
        all_results = []
        with open(output_path, 'r') as f:
            for line in f:
                all_results.append(json.loads(line))

        # Breakdown by type
        type_scores = {}
        for r in all_results:
            if r['parsed_scores'] and 'total_score' in r['parsed_scores']:
                t = r['type']
                if t not in type_scores:
                    type_scores[t] = []
                type_scores[t].append(r['parsed_scores']['total_score'])

        print(f"\nBreakdown by difficulty:")
        for t in sorted(type_scores.keys()):
            scores = type_scores[t]
            avg = sum(scores) / len(scores)
            print(f"  {t:15s}: {avg:.2f}/10 ({len(scores)} samples)")

        # Save summary
        summary = {
            'total_processed': len(successful_results),
            'successfully_scored': len(total_scores),
            'image_url_failures': image_url_failures,
            'gpt4o_failures': gpt4o_failures,
            'parse_failures': parse_failures,
            'average_score': avg_score,
            'max_score': max(total_scores),
            'min_score': min(total_scores),
            'type_breakdown': {t: sum(scores)/len(scores) for t, scores in type_scores.items()}
        }

        summary_path = "/u/hli36/MPU-RL-clean/src/multi-agent/benchmark_evaluation/llava/outputs/mia/eval_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")
    else:
        print("\n✗ No scores were successfully parsed!")
        print(f"\nFailure breakdown:")
        print(f"  Image URL failures: {image_url_failures}")
        print(f"  GPT-4o call failures: {gpt4o_failures}")
        print(f"  Parse failures: {parse_failures}")

if __name__ == "__main__":
    main()
