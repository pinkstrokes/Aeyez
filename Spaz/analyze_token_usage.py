#!/usr/bin/env python3
"""
Token Usage Analysis Tool

Analyzes token usage and costs from JSON log files.
Usage: python analyze_token_usage.py <log_directory>
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

qwen25_vl_3b_INPUT_TOKEN_PRICE_PER_1K = 0.00021 
qwen25_vl_3b_OUTPUT_TOKEN_PRICE_PER_1K = 0.00063
qwen25_vl_32b_INPUT_TOKEN_PRICE_PER_1K = 0.00140
qwen25_vl_32b_OUTPUT_TOKEN_PRICE_PER_1K = 0.00420
qwen3_8b_INPUT_TOKEN_PRICE_PER_1K = 0.00018
qwen3_8b_OUTPUT_TOKEN_PRICE_PER_1K = 0.00070

# Pricing constants by model (update these as needed)
MODEL_PRICING = {
    'translator': {
        'input_price_per_1k': qwen25_vl_3b_INPUT_TOKEN_PRICE_PER_1K,    # Vision model pricing
        'output_price_per_1k': qwen25_vl_3b_OUTPUT_TOKEN_PRICE_PER_1K
    },
    'reasoning': {
        'input_price_per_1k': qwen3_8b_INPUT_TOKEN_PRICE_PER_1K,  # Text-only model pricing
        'output_price_per_1k': qwen3_8b_OUTPUT_TOKEN_PRICE_PER_1K 
    }
}


# Default pricing for unknown models
DEFAULT_INPUT_PRICE_PER_1K = 0.003
DEFAULT_OUTPUT_PRICE_PER_1K = 0.015


def find_all_token_usage(obj, token_usages: List[Dict]):
    """Recursively find all token_usage entries in a nested object."""
    if isinstance(obj, dict):
        if 'token_usage' in obj:
            token_usage = obj['token_usage']
            if isinstance(token_usage, dict):
                token_usages.append(token_usage)

        # Also check if the object itself looks like a token_usage dict
        if 'input_tokens' in obj and 'completion_tokens' in obj:
            token_usages.append(obj)

        for value in obj.values():
            find_all_token_usage(value, token_usages)
    elif isinstance(obj, list):
        for item in obj:
            find_all_token_usage(item, token_usages)


def extract_token_usage(json_file: Path) -> Dict:
    """Extract all token usage data from a JSON log file with agent-specific breakdown."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check for different token usage field names (prioritize question-specific usage)
        token_usage_field = None
        if 'token_usage_this_question' in data:
            token_usage_field = 'token_usage_this_question'
        elif 'token_usage' in data:
            token_usage_field = 'token_usage'

        # Check for agent-specific token usage first (top-level structure)
        agent_usage = {}
        if token_usage_field and 'agents' in data[token_usage_field]:
            agents_data = data[token_usage_field]['agents']
            for agent_name, usage in agents_data.items():
                agent_usage[agent_name] = {
                    'input_tokens': usage.get('input_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0)
                }

        # If we have agent-specific data, use that; otherwise fall back to recursive search
        if agent_usage:
            total_input = 0
            total_output = 0
            total_cost = 0.0
            total_input_cost = 0.0
            total_output_cost = 0.0

            for agent_name, usage in agent_usage.items():
                total_input += usage['input_tokens']
                total_output += usage['completion_tokens']

                # Calculate cost using agent-specific pricing
                pricing = MODEL_PRICING.get(agent_name, {
                    'input_price_per_1k': DEFAULT_INPUT_PRICE_PER_1K,
                    'output_price_per_1k': DEFAULT_OUTPUT_PRICE_PER_1K
                })

                input_cost = (usage['input_tokens'] / 1000) * pricing['input_price_per_1k']
                output_cost = (usage['completion_tokens'] / 1000) * pricing['output_price_per_1k']
                total_input_cost += input_cost
                total_output_cost += output_cost
                total_cost += input_cost + output_cost

            return {
                'file': json_file.name,
                'total_input_tokens': total_input,
                'total_completion_tokens': total_output,
                'total_tokens': total_input + total_output,
                'agent_breakdown': agent_usage,
                'total_cost': total_cost,
                'input_cost': total_input_cost,
                'output_cost': total_output_cost,
                'num_agents': len(agent_usage)
            }

        else:
            # Fall back to recursive search
            all_token_usages = []
            find_all_token_usage(data, all_token_usages)

            if not all_token_usages:
                return None

            # Sum up all token usage
            total_input = 0
            total_output = 0
            total_tokens = 0

            for token_usage in all_token_usages:
                total_input += token_usage.get('input_tokens', 0)
                total_output += token_usage.get('completion_tokens', 0)
                total_tokens += token_usage.get('total_tokens', 0)

            # Calculate cost using default pricing
            input_cost = (total_input / 1000) * DEFAULT_INPUT_PRICE_PER_1K
            output_cost = (total_output / 1000) * DEFAULT_OUTPUT_PRICE_PER_1K
            total_cost = input_cost + output_cost

            return {
                'file': json_file.name,
                'total_input_tokens': total_input,
                'total_completion_tokens': total_output,
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'num_token_usage_entries': len(all_token_usages)
            }

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {json_file}: {e}")
        return None


def calculate_cost(input_tokens: int, output_tokens: int, agent_name: str = None) -> float:
    """Calculate cost based on token usage and agent type."""
    if agent_name and agent_name in MODEL_PRICING:
        pricing = MODEL_PRICING[agent_name]
        input_cost = (input_tokens / 1000) * pricing['input_price_per_1k']
        output_cost = (output_tokens / 1000) * pricing['output_price_per_1k']
    else:
        input_cost = (input_tokens / 1000) * DEFAULT_INPUT_PRICE_PER_1K
        output_cost = (output_tokens / 1000) * DEFAULT_OUTPUT_PRICE_PER_1K
    return input_cost + output_cost


def analyze_directory(log_dir: Path, file_limit: int = None) -> Tuple[List[Dict], Dict]:
    """Analyze JSON files in a directory with optional file limit."""
    json_files = list(log_dir.glob('**/*.json'))

    if not json_files:
        print(f"No JSON files found in {log_dir}")
        return [], {}

    # Apply file limit if specified
    if file_limit is not None and file_limit > 0:
        json_files = json_files[:file_limit]
        print(f"Processing {len(json_files)} of {len(list(log_dir.glob('**/*.json')))} JSON files in {log_dir}")
    else:
        print(f"Found {len(json_files)} JSON files in {log_dir}")

    usage_data = []
    total_input = 0
    total_output = 0
    total_cost = 0.0
    total_input_cost = 0.0
    total_output_cost = 0.0
    processed_files = 0

    # Agent-wise totals
    agent_totals = {}

    for json_file in json_files:
        file_usage = extract_token_usage(json_file)
        if file_usage:
            usage_data.append(file_usage)
            total_input += file_usage['total_input_tokens']
            total_output += file_usage['total_completion_tokens']
            total_cost += file_usage['total_cost']
            total_input_cost += file_usage.get('input_cost', 0.0)
            total_output_cost += file_usage.get('output_cost', 0.0)
            processed_files += 1

            # Track agent-wise totals
            if 'agent_breakdown' in file_usage:
                for agent_name, agent_data in file_usage['agent_breakdown'].items():
                    if agent_name not in agent_totals:
                        agent_totals[agent_name] = {'input': 0, 'output': 0, 'cost': 0.0}

                    agent_totals[agent_name]['input'] += agent_data['input_tokens']
                    agent_totals[agent_name]['output'] += agent_data['completion_tokens']
                    agent_totals[agent_name]['cost'] += calculate_cost(
                        agent_data['input_tokens'],
                        agent_data['completion_tokens'],
                        agent_name
                    )

    # Calculate averages
    if processed_files > 0:
        avg_input = total_input / processed_files
        avg_output = total_output / processed_files
        avg_cost = total_cost / processed_files
        avg_input_cost = total_input_cost / processed_files
        avg_output_cost = total_output_cost / processed_files
    else:
        avg_input = avg_output = avg_cost = 0
        avg_input_cost = avg_output_cost = 0.0

    # Calculate median and trimmed statistics
    median_input = median_output = median_cost = None
    trimmed_avg_input = trimmed_avg_output = trimmed_avg_cost = None
    trimmed_count = 0

    if usage_data:
        # Extract all values for calculations
        all_inputs = [x['total_input_tokens'] for x in usage_data]
        all_outputs = [x['total_completion_tokens'] for x in usage_data]
        all_costs = [x.get('total_cost', 0) for x in usage_data]

        # Calculate medians
        all_inputs_sorted = sorted(all_inputs)
        all_outputs_sorted = sorted(all_outputs)
        all_costs_sorted = sorted(all_costs)

        n = len(all_inputs_sorted)
        if n > 0:
            if n % 2 == 0:
                median_input = (all_inputs_sorted[n//2 - 1] + all_inputs_sorted[n//2]) / 2
                median_output = (all_outputs_sorted[n//2 - 1] + all_outputs_sorted[n//2]) / 2
                median_cost = (all_costs_sorted[n//2 - 1] + all_costs_sorted[n//2]) / 2
            else:
                median_input = all_inputs_sorted[n//2]
                median_output = all_outputs_sorted[n//2]
                median_cost = all_costs_sorted[n//2]

            # Calculate trimmed mean (exclude top/bottom 5%)
            if n >= 20:  # Only if we have enough data points
                trim_count = max(1, int(0.05 * n))  # 5% from each end
                trimmed_inputs = all_inputs_sorted[trim_count:-trim_count]
                trimmed_outputs = all_outputs_sorted[trim_count:-trim_count]
                trimmed_costs = all_costs_sorted[trim_count:-trim_count]

                if trimmed_inputs:
                    trimmed_avg_input = sum(trimmed_inputs) / len(trimmed_inputs)
                    trimmed_avg_output = sum(trimmed_outputs) / len(trimmed_outputs)
                    trimmed_avg_cost = sum(trimmed_costs) / len(trimmed_costs)
                    trimmed_count = len(trimmed_inputs)

    # Get top 5 files by cost for summary
    top_files = sorted(usage_data, key=lambda x: x.get('total_cost', 0), reverse=True)[:5]

    summary = {
        'total_files': len(json_files),
        'processed_files': processed_files,
        'total_input_tokens': total_input,
        'total_output_tokens': total_output,
        'total_cost': total_cost,
        'total_input_cost': total_input_cost,
        'total_output_cost': total_output_cost,
        'avg_input_tokens': avg_input,
        'avg_output_tokens': avg_output,
        'avg_cost_per_file': avg_cost,
        'avg_input_cost': avg_input_cost,
        'avg_output_cost': avg_output_cost,
        'median_input_tokens': median_input,
        'median_output_tokens': median_output,
        'median_cost_per_file': median_cost,
        'trimmed_avg_input_tokens': trimmed_avg_input,
        'trimmed_avg_output_tokens': trimmed_avg_output,
        'trimmed_avg_cost_per_file': trimmed_avg_cost,
        'trimmed_file_count': trimmed_count,
        'agent_totals': agent_totals,
        'model_pricing': MODEL_PRICING,
        'top_files': top_files
    }

    return usage_data, summary


def print_summary(summary: Dict):
    """Print a formatted summary of the analysis."""
    print("\n" + "="*70)
    print("TOKEN USAGE ANALYSIS SUMMARY")
    print("="*70)

    print(f"Files found: {summary['total_files']}")
    print(f"Files processed: {summary['processed_files']}")
    print()

    print("TOTAL USAGE:")
    print(f"  Input tokens:  {summary['total_input_tokens']:,}")
    print(f"  Output tokens: {summary['total_output_tokens']:,}")
    print(f"  Input cost:    ${summary['total_input_cost']:.4f}")
    print(f"  Output cost:   ${summary['total_output_cost']:.4f}")
    print(f"  Total cost:    ${summary['total_cost']:.4f}")
    print()

    print("AVERAGE PER FILE:")
    print(f"  Input tokens:  {summary['avg_input_tokens']:,.1f}")
    print(f"  Output tokens: {summary['avg_output_tokens']:,.1f}")
    print(f"  Input cost:    ${summary['avg_input_cost']:.4f}")
    print(f"  Output cost:   ${summary['avg_output_cost']:.4f}")
    print(f"  Total cost:    ${summary['avg_cost_per_file']:.4f}")
    print()

    # Add median and trimmed mean statistics
    if summary.get('median_input_tokens') is not None:
        print("MEDIAN PER FILE:")
        print(f"  Input tokens:  {summary['median_input_tokens']:,.1f}")
        print(f"  Output tokens: {summary['median_output_tokens']:,.1f}")
        print(f"  Cost per file: ${summary['median_cost_per_file']:.4f}")
        print()

    if summary.get('trimmed_avg_input_tokens') is not None:
        print("TRIMMED AVERAGE (excludes top/bottom 5%):")
        print(f"  Input tokens:  {summary['trimmed_avg_input_tokens']:,.1f}")
        print(f"  Output tokens: {summary['trimmed_avg_output_tokens']:,.1f}")
        print(f"  Cost per file: ${summary['trimmed_avg_cost_per_file']:.4f}")
        print(f"  Files used:    {summary['trimmed_file_count']} of {summary['processed_files']}")
        print()

    # Agent-specific breakdown
    if summary.get('agent_totals'):
        print("BREAKDOWN BY AGENT:")
        for agent_name, totals in summary['agent_totals'].items():
            print(f"  {agent_name}:")
            print(f"    Input tokens:  {totals['input']:,}")
            print(f"    Output tokens: {totals['output']:,}")
            print(f"    Cost:          ${totals['cost']:.4f}")
        print()

    # Model pricing configuration
    if summary.get('model_pricing'):
        print("PRICING CONFIGURATION:")
        for agent_name, pricing in summary['model_pricing'].items():
            print(f"  {agent_name}:")
            print(f"    Input:  ${pricing['input_price_per_1k']:.5f} per 1K tokens")
            print(f"    Output: ${pricing['output_price_per_1k']:.5f} per 1K tokens")
        print()

    # Show top 5 processed files
    if summary.get('top_files'):
        print("TOP 5 PROCESSED FILES BY COST:")
        for i, file_data in enumerate(summary['top_files'], 1):
            cost = file_data.get('total_cost', 0)
            print(f"  {i}. {file_data['file']:<25} - ${cost:>7.4f} "
                  f"({file_data['total_input_tokens']:,} in, {file_data['total_completion_tokens']:,} out)")
        print()

    print("="*70)


def print_percentile_analysis(usage_data: List[Dict]):
    """Print percentile analysis to identify outliers."""
    if not usage_data:
        return

    # Extract metrics for analysis
    input_tokens = [x['total_input_tokens'] for x in usage_data]
    output_tokens = [x['total_completion_tokens'] for x in usage_data]
    costs = [x.get('total_cost', 0) for x in usage_data]

    # Sort for percentile calculation
    input_sorted = sorted(input_tokens)
    output_sorted = sorted(output_tokens)
    cost_sorted = sorted(costs)

    def get_percentile(data, percentile):
        """Calculate percentile value."""
        if not data:
            return 0
        idx = int((percentile / 100.0) * (len(data) - 1))
        return data[idx]

    def get_percentile_range(data, start_pct, end_pct):
        """Get values in percentile range."""
        if not data:
            return []
        start_idx = int((start_pct / 100.0) * len(data))
        end_idx = int((end_pct / 100.0) * len(data))
        return data[start_idx:end_idx]

    print(f"\nPERCENTILE DISTRIBUTION ANALYSIS ({len(usage_data)} files):")
    print("="*80)

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    print(f"{'Percentile':<12} {'Input Tokens':<15} {'Output Tokens':<16} {'Cost ($)':<10}")
    print("-"*80)

    for p in percentiles:
        input_val = get_percentile(input_sorted, p)
        output_val = get_percentile(output_sorted, p)
        cost_val = get_percentile(cost_sorted, p)
        print(f"{p:>3}%         {input_val:<15,} {output_val:<16,} ${cost_val:<9.4f}")

    # Show ranges and their contribution
    print(f"\nRANGE ANALYSIS:")
    print("-"*80)

    ranges = [
        ("Top 1%", 99, 100),
        ("Top 5%", 95, 100),
        ("Top 10%", 90, 100),
        ("Top 30%", 70, 100),
        ("Top 50%", 50, 100),
        ("Bottom 50%", 0, 50)
    ]

    total_input = sum(input_tokens)
    total_output = sum(output_tokens)
    total_cost = sum(costs)

    print(f"{'Range':<12} {'Files':<8} {'Input %':<10} {'Output %':<11} {'Cost %':<10} {'Avg Input':<12} {'Avg Output':<12}")
    print("-"*80)

    # Sort usage_data by cost for getting actual files in ranges
    usage_sorted_by_cost = sorted(usage_data, key=lambda x: x.get('total_cost', 0), reverse=True)

    for range_name, start_pct, end_pct in ranges:
        # Get files in this range
        num_files_in_range = int((end_pct - start_pct) / 100.0 * len(usage_data))
        if num_files_in_range == 0:
            continue

        # Get the actual data for this range
        start_idx = int((start_pct / 100.0) * len(usage_data))
        end_idx = int((end_pct / 100.0) * len(usage_data))

        range_input = input_sorted[start_idx:end_idx] if end_idx > start_idx else input_sorted[start_idx:]
        range_output = output_sorted[start_idx:end_idx] if end_idx > start_idx else output_sorted[start_idx:]
        range_cost = cost_sorted[start_idx:end_idx] if end_idx > start_idx else cost_sorted[start_idx:]

        if range_input:
            input_pct = sum(range_input) / total_input * 100
            output_pct = sum(range_output) / total_output * 100
            cost_pct = sum(range_cost) / total_cost * 100
            avg_input = sum(range_input) / len(range_input)
            avg_output = sum(range_output) / len(range_output)

            print(f"{range_name:<12} {len(range_input):<8} {input_pct:<9.1f}% {output_pct:<10.1f}% {cost_pct:<9.1f}% {avg_input:<12,.0f} {avg_output:<12,.0f}")

            # Show sample files from this range (by cost ranking)
            # For ranges like "Top 1%" we want the highest cost files
            # For ranges like "Bottom 50%" we want the lowest cost files

            if range_name == "Top 1%":
                # Top 1% = highest 1% by cost
                sample_files = usage_sorted_by_cost[:2]  # Top 1% of 123 = ~1.2 files
            elif range_name == "Top 5%":
                # Top 5% = highest 5% by cost
                sample_files = usage_sorted_by_cost[:6]  # Top 5% of 123 = ~6 files
            elif range_name == "Top 10%":
                # Top 10% = highest 10% by cost
                sample_files = usage_sorted_by_cost[:12]  # Top 10% of 123 = ~12 files
            elif range_name == "Top 30%":
                # Top 30% = highest 30% by cost
                sample_files = usage_sorted_by_cost[:37]  # Top 30% of 123 = ~37 files
            elif range_name == "Top 50%":
                # Top 50% = highest 50% by cost
                sample_files = usage_sorted_by_cost[:62]  # Top 50% of 123 = ~62 files
            elif range_name == "Bottom 50%":
                # Bottom 50% = lowest 50% by cost
                sample_files = usage_sorted_by_cost[62:]  # Bottom 50% = remaining files

            # Show up to 5 sample files
            sample_files = sample_files[:5]
            if sample_files:
                file_names = [f['file'] for f in sample_files]
                print(f"             Sample files: {', '.join(file_names)}")

    print("="*80)


def print_detailed_breakdown(usage_data: List[Dict], limit: int = 10):
    """Print detailed breakdown for top files by cost."""
    if not usage_data:
        return

    # Sort by total cost (descending)
    sorted_data = sorted(usage_data,
                        key=lambda x: x.get('total_cost', 0),
                        reverse=True)

    print(f"\nTOP {min(limit, len(sorted_data))} FILES BY COST:")
    print("-" * 100)
    print(f"{'File':<30} {'Input':<10} {'Output':<10} {'Agents':<8} {'Cost':<10}")
    print("-" * 100)

    for i, file_data in enumerate(sorted_data[:limit]):
        cost = file_data.get('total_cost', 0)
        agents_info = file_data.get('num_agents', file_data.get('num_token_usage_entries', 'N/A'))
        print(f"{file_data['file']:<30} "
              f"{file_data['total_input_tokens']:<10,} "
              f"{file_data['total_completion_tokens']:<10,} "
              f"{agents_info:<8} "
              f"${cost:<9.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze token usage in JSON log files")
    parser.add_argument('log_directory', help='Directory containing JSON log files')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed breakdown of top files')
    parser.add_argument('--percentiles', '-p', action='store_true',
                       help='Show percentile distribution analysis')
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='Limit number of files to process (default: process all files). For detailed view, shows this many top files.')
    parser.add_argument('--detailed-limit', type=int, default=10,
                       help='Number of files to show in detailed view (default: 10)')

    args = parser.parse_args()

    log_dir = Path(args.log_directory)

    if not log_dir.exists():
        print(f"Error: Directory '{log_dir}' does not exist")
        sys.exit(1)

    if not log_dir.is_dir():
        print(f"Error: '{log_dir}' is not a directory")
        sys.exit(1)

    # Analyze the directory with optional file limit
    usage_data, summary = analyze_directory(log_dir, args.limit)

    # Print results
    print_summary(summary)

    if args.percentiles and usage_data:
        print_percentile_analysis(usage_data)

    if args.detailed and usage_data:
        detailed_limit = args.detailed_limit if args.limit is None else min(args.limit, args.detailed_limit)
        print_detailed_breakdown(usage_data, detailed_limit)


if __name__ == "__main__":
    main()