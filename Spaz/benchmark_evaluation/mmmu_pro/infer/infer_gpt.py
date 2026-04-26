#!/usr/bin/env python3
"""
MMMU PRO inference script for GPT models.
Usage: python infer/infer_gpt.py [MODEL_NAME] [MODE] [SETTING]

Example:
    python infer/infer_gpt.py gpt-4o-mini cot vision
    python infer/infer_gpt.py gpt4o-mini direct "standard (10 options)"
"""

import os
import sys
import asyncio
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_utils import MMMUProDataset, format_options
from model_utils import create_model_interface, MODEL_MAPPINGS, CONFIG_MAPPINGS
from prompt_utils import MMMUProPromptManager, extract_answer_from_response


class MMMUProInferenceGPT:
    """MMMU PRO inference engine for GPT models."""
    
    def __init__(self, model_name: str, mode: str, setting: str):
        """Initialize inference engine.
        
        Args:
            model_name: Name of the GPT model
            mode: Inference mode ('cot' or 'direct')
            setting: Dataset setting ('vision', 'standard (4 options)', 'standard (10 options)')
        """
        self.model_name = model_name
        self.mode = mode
        self.setting = setting
        
        # Normalize model name
        if model_name == "gpt4o-mini":
            model_name = "gpt-4o-mini"
        
        # Validate parameters
        if model_name not in MODEL_MAPPINGS:
            raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_MAPPINGS.keys())}")
        
        if mode not in ['cot', 'direct']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'cot' or 'direct'")
        
        # Map model name to API model name and config name
        self.api_model_name = MODEL_MAPPINGS[model_name]
        self.config_name = CONFIG_MAPPINGS[model_name]
        
        # Initialize dataset
        self.dataset = MMMUProDataset()
        
        # Initialize prompt manager
        self.prompt_manager = MMMUProPromptManager()
        
        # Initialize model interface
        try:
            self.model = create_model_interface(self.api_model_name, self.config_name)
            print(f"✓ Model interface initialized: {model_name} -> {self.config_name}")
        except Exception as e:
            print(f"✗ Failed to initialize model interface: {e}")
            raise
            
        # Create output directory
        self.output_dir = "./output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def build_prompt(self, sample: Dict[str, Any]) -> Tuple[str, List]:
        """Build prompt for the model based on mode and setting.
        
        Args:
            sample: Processed sample from dataset
            
        Returns:
            Tuple of (formatted_prompt_string, ordered_images)
        """
        return self.prompt_manager.build_prompt(sample, self.mode, self.setting)
    
    def extract_answer(self, response: str, sample: Dict[str, Any]) -> str:
        """Extract answer from model response.
        
        Args:
            response: Raw model response
            sample: Sample data for context
            
        Returns:
            Extracted answer
        """
        return extract_answer_from_response(
            response, 
            self.setting, 
            sample.get('options', [])
        )
    
    async def infer_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a single sample.
        
        Args:
            sample: Processed sample from dataset
            
        Returns:
            Result dictionary
        """
        prompt, ordered_images = self.build_prompt(sample)
        
        try:
            # Record token usage before this sample (for calculating delta)
            token_before = None
            if hasattr(self.model, 'token_tracker'):
                token_before = self.model.token_tracker.get_usage_summary().copy()
            
            # Generate response with properly ordered images
            start_time = time.time()
            response = await self.model.generate_response(prompt, ordered_images)
            inference_time = time.time() - start_time
            
            # Extract answer
            extracted_answer = self.extract_answer(response, sample)
            
            # Calculate token usage for this sample (delta from before)
            token_usage = {}
            if hasattr(self.model, 'token_tracker'):
                token_after = self.model.token_tracker.get_usage_summary()
                if token_before:
                    token_usage = {
                        'input_tokens': token_after['input_tokens'] - token_before['input_tokens'],
                        'completion_tokens': token_after['completion_tokens'] - token_before['completion_tokens'],
                        'total_tokens': token_after['total_tokens'] - token_before['total_tokens']
                    }
                else:
                    token_usage = token_after

            # Build result
            result = {
                'id': sample['id'],
                'subset': sample['subset'],
                'subject': sample['subject'],
                'topic_difficulty': sample.get('topic_difficulty', ''),
                'question': sample['question'],
                'options': sample.get('options', []),
                'ground_truth': sample['answer'],
                'model_response': response,
                'extracted_answer': extracted_answer,
                'inference_time': inference_time,
                'model_name': self.model_name,
                'mode': self.mode,
                'setting': self.setting,
                'timestamp': datetime.now().isoformat(),
                'token_usage': token_usage
            }

            return result
            
        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")
            return {
                'id': sample['id'],
                'subset': sample['subset'],
                'subject': sample['subject'],
                'error': str(e),
                'model_name': self.model_name,
                'mode': self.mode,
                'setting': self.setting,
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_inference(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """Run inference on the dataset.
        
        Args:
            max_samples: Maximum number of samples to process
            
        Returns:
            List of result dictionaries
        """
        print(f"Starting MMMU PRO inference...")
        print(f"Model: {self.model_name}")
        print(f"Mode: {self.mode}")
        print(f"Setting: {self.setting}")
        print(f"Max samples: {max_samples or 'All'}")
        
        # Load dataset
        datasets = self.dataset.load_subsets([self.setting])
        samples = self.dataset.get_samples(self.setting, max_samples, validation_only=True)
        
        print(f"Processing {len(samples)} samples...")
        
        results = []
        
        # Process samples with progress bar
        for sample in tqdm(samples, desc="Inference"):
            processed_sample = self.dataset.process_sample(sample, self.setting)
            result = await self.infer_single_sample(processed_sample)
            results.append(result)
            
            # Add delay to respect rate limits
            await asyncio.sleep(0.5)  # GPT models may have stricter rate limits
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to output directory.
        
        Args:
            results: List of result dictionaries
        """
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mmmu_pro_{self.model_name.replace('-', '_')}_{self.mode}_{self.setting.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.jsonl"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Results saved to: {output_path}")
        
        # Also save a summary
        summary = {
            'model_name': self.model_name,
            'mode': self.mode,
            'setting': self.setting,
            'total_samples': len(results),
            'successful_samples': len([r for r in results if 'error' not in r]),
            'failed_samples': len([r for r in results if 'error' in r]),
            'output_file': filename,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = output_path.replace('.jsonl', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Summary saved to: {summary_path}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='MMMU PRO inference for GPT models')
    parser.add_argument('model_name', help='Model name (e.g., gpt-4o-mini, gpt4o-mini)')
    parser.add_argument('mode', choices=['cot', 'direct'], help='Inference mode')
    parser.add_argument('setting', help='Dataset setting (e.g., vision, "standard (10 options)")')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        inference = MMMUProInferenceGPT(args.model_name, args.mode, args.setting)
    except Exception as e:
        print(f"Failed to initialize inference engine: {e}")
        return 1
    
    # Run inference
    try:
        results = await inference.run_inference(args.max_samples)
        inference.save_results(results)
        print(f"✓ Inference completed successfully!")
        return 0
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
