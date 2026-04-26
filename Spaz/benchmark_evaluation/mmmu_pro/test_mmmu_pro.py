#!/usr/bin/env python3
"""
Test script for MMMU PRO implementation.
Tests dataset loading, prompt generation, and basic functionality.
"""

import os
import sys
import asyncio
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_utils import MMMUProDataset
from prompt_utils import MMMUProPromptManager, extract_answer_from_response


async def test_dataset_loading():
    """Test dataset loading functionality."""
    print("🧪 Testing dataset loading...")
    
    try:
        dataset = MMMUProDataset()
        
        # Test loading each subset with 1 sample
        for subset in ["vision", "standard (4 options)", "standard (10 options)"]:
            print(f"\nTesting subset: {subset}")
            datasets = dataset.load_subsets([subset])
            samples = dataset.get_samples(subset, max_samples=1)
            
            if samples:
                sample = samples[0]
                processed = dataset.process_sample(sample, subset)
                print(f"✓ Sample ID: {processed['id']}")
                print(f"✓ Subject: {processed['subject']}")
                print(f"✓ Images: {len(processed['images'])}")
                print(f"✓ Question: {processed['question'][:100]}...")
                if processed['options']:
                    print(f"✓ Options: {len(processed['options'])} choices")
                
                # Test prompt generation
                prompt_manager = MMMUProPromptManager()
                for mode in ["cot", "direct"]:
                    prompt, ordered_images = prompt_manager.build_prompt(processed, mode, subset)
                    print(f"✓ {mode.upper()} prompt generated: {len(prompt)} chars")
                    print(f"✓ Ordered images: {len(ordered_images)}")
            else:
                print(f"✗ No samples found for {subset}")
                
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()


def test_prompt_generation():
    """Test prompt generation with official templates."""
    print("\n🧪 Testing prompt generation...")
    
    prompt_manager = MMMUProPromptManager()
    
    # Test sample data
    test_samples = [
        {
            'id': 'test_vision_1',
            'subset': 'vision',
            'question': '',
            'images': ['dummy_image'],
            'options': [],
            'answer': 'A'
        },
        {
            'id': 'test_standard_1',
            'subset': 'standard (10 options)',
            'question': 'What is 2+2?',
            'images': ['dummy_image1', 'dummy_image2'],
            'options': ['<image 2>', '<image 1>', '4', '5', '6', '7', '8', '9', '10', '11'],
            'answer': 'C'
        }
    ]
    
    for sample in test_samples:
        print(f"\n--- Testing sample: {sample['id']} ---")
        for mode in ["cot", "direct"]:
            try:
                prompt, ordered_images = prompt_manager.build_prompt(sample, mode, sample['subset'])
                print(f"✓ {mode.upper()} mode:")
                print(f"  Prompt: {prompt[:200]}...")
                print(f"  Images: {len(ordered_images)}")
            except Exception as e:
                print(f"✗ {mode.upper()} mode failed: {e}")


def test_answer_extraction():
    """Test answer extraction functionality."""
    print("\n🧪 Testing answer extraction...")
    
    test_cases = [
        # Official format
        ("I think the answer is B. Answer: B", "standard (10 options)", ["A", "B", "C", "D"], "B"),
        ("Let me think... Answer: C", "vision", [], "C"),
        
        # Alternative formats
        ("The correct answer is A.", "standard (4 options)", ["A", "B", "C", "D"], "A"),
        ("I choose option D", "standard (10 options)", ["A", "B", "C", "D", "E"], "D"),
        
        # Edge cases
        ("This is a complex question. After analysis, I believe A is correct.", "standard (4 options)", ["A", "B", "C", "D"], "A"),
    ]
    
    for response, setting, options, expected in test_cases:
        extracted = extract_answer_from_response(response, setting, options)
        status = "✓" if extracted == expected else "✗"
        print(f"{status} '{response[:50]}...' -> '{extracted}' (expected: '{expected}')")


async def main():
    """Main test function."""
    print("🚀 MMMU PRO Implementation Test")
    print("=" * 50)
    
    # Test 1: Dataset loading
    await test_dataset_loading()
    
    # Test 2: Prompt generation
    test_prompt_generation()
    
    # Test 3: Answer extraction
    test_answer_extraction()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
