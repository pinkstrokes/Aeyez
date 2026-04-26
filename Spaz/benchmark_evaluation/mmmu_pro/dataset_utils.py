"""
Dataset utilities for MMMU PRO benchmark.
Handles loading and processing of MMMU PRO datasets from Hugging Face.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import pandas as pd
from PIL import Image
import requests
from io import BytesIO


class MMMUProDataset:
    """MMMU PRO dataset loader and processor."""
    
    AVAILABLE_SUBSETS = [
        "standard (4 options)",
        "standard (10 options)", 
        "vision"
    ]
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize MMMU PRO dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        import tempfile
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="mmmu_pro_cache_")
        self.datasets = {}
        
    def load_subsets(self, subsets: List[str] = None) -> Dict[str, Any]:
        """Load specified MMMU PRO subsets.
        
        Args:
            subsets: List of subset names to load. If None, loads all subsets.
            
        Returns:
            Dictionary mapping subset names to loaded datasets
        """
        if subsets is None:
            subsets = self.AVAILABLE_SUBSETS
            
        # Validate subset names
        invalid_subsets = [s for s in subsets if s not in self.AVAILABLE_SUBSETS]
        if invalid_subsets:
            raise ValueError(f"Invalid subsets: {invalid_subsets}. "
                           f"Available: {self.AVAILABLE_SUBSETS}")
        
        print(f"Loading MMMU PRO subsets: {subsets}")
        
        for subset in subsets:
            print(f"Loading subset: {subset}")
            try:
                dataset = load_dataset(
                    "MMMU/MMMU_Pro", 
                    subset, 
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                self.datasets[subset] = dataset
                print(f"✓ Loaded {subset}: {len(dataset['test'])} samples")
            except Exception as e:
                print(f"✗ Failed to load {subset}: {e}")
                raise
                
        return self.datasets
    
    def get_samples(self, subset: str, max_samples: Optional[int] = None, validation_only: bool = False) -> List[Dict[str, Any]]:
        """Get samples from a specific subset.
        
        Args:
            subset: Name of the subset
            max_samples: Maximum number of samples to return
            validation_only: If True, only return samples with validation_ prefix
            
        Returns:
            List of sample dictionaries
        """
        if subset not in self.datasets:
            raise ValueError(f"Subset {subset} not loaded. Available: {list(self.datasets.keys())}")
            
        samples = list(self.datasets[subset]['test'])
        
        # Filter for validation samples only if requested
        if validation_only:
            samples = [s for s in samples if s.get('id', '').startswith('validation_')]
            print(f"Filtered to validation samples: {len(samples)} samples")
        
        if max_samples is not None:
            samples = samples[:max_samples]
            
        print(f"Retrieved {len(samples)} samples from {subset}")
        return samples
    
    def process_sample(self, sample: Dict[str, Any], subset: str) -> Dict[str, Any]:
        """Process a single sample for inference.
        
        Args:
            sample: Raw sample from dataset
            subset: Name of the subset
            
        Returns:
            Processed sample ready for model inference
        """
        processed = {
            'id': sample.get('id', ''),
            'subset': subset,
            'subject': sample.get('subject', ''),
            'topic_difficulty': sample.get('topic_difficulty', ''),
        }
        
        if subset == "vision":
            # Vision subset: question embedded in image, but options are available
            processed['question'] = ""  # No separate text question in vision mode
            processed['images'] = [sample['image']]
            processed['answer'] = sample['answer']
            
            # Parse options - they are available in vision mode
            options = sample.get('options', [])
            if isinstance(options, str):
                # Try to parse string representation of list
                try:
                    import ast
                    processed['options'] = ast.literal_eval(options)
                except (ValueError, SyntaxError):
                    # If parsing fails, treat as single option or split by newlines
                    processed['options'] = [options] if options else []
            else:
                processed['options'] = options if options else []
            
        else:
            # Standard subsets: separate question and options
            processed['question'] = sample['question']
            
            # Parse options - they might be stored as a string representation of a list
            options = sample['options']
            if isinstance(options, str):
                # Try to parse string representation of list
                try:
                    import ast
                    processed['options'] = ast.literal_eval(options)
                except (ValueError, SyntaxError):
                    # If parsing fails, treat as single option or split by newlines
                    processed['options'] = [options] if options else []
            else:
                processed['options'] = options
                
            processed['answer'] = sample['answer']
            processed['explanation'] = sample.get('explanation', '')
            
            # Handle multiple images
            images = []
            for i in range(1, 8):  # Check image_1 to image_7
                img_key = f'image_{i}'
                if img_key in sample and sample[img_key] is not None:
                    images.append(sample[img_key])
            processed['images'] = images
            
        return processed
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save inference results to file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"Results saved to: {output_path}")


def parse_subsets_arg(subsets_str: str) -> List[str]:
    """Parse comma-separated subsets string.
    
    Args:
        subsets_str: Comma-separated string of subset names
        
    Returns:
        List of subset names
    """
    if subsets_str.lower() == "all":
        return MMMUProDataset.AVAILABLE_SUBSETS
    
    subsets = [s.strip() for s in subsets_str.split(',')]
    return subsets


def format_options(options: List[str]) -> str:
    """Format options for display in prompts.
    
    Args:
        options: List of option strings
        
    Returns:
        Formatted options string
    """
    if not options:
        return ""
        
    formatted = []
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)
        formatted.append(f"{letter}. {option}")
        
    return "\n".join(formatted)


if __name__ == "__main__":
    # Test the dataset loader
    dataset = MMMUProDataset()
    datasets = dataset.load_subsets(["vision"])
    
    samples = dataset.get_samples("vision", max_samples=2)
    for sample in samples:
        processed = dataset.process_sample(sample, "vision")
        print(f"Sample ID: {processed['id']}")
        print(f"Subject: {processed['subject']}")
        print(f"Images: {len(processed['images'])}")
        print("---")
