"""
Prompt utilities for MMMU PRO benchmark.
Handles official prompt templates and image token processing.
"""

import re
from typing import List, Dict, Any, Tuple
from PIL import Image


class MMMUProPromptManager:
    """Manages MMMU PRO official prompts and image token processing."""
    
    # Official MMMU PRO prompts (from official prompts.yaml)
    OFFICIAL_PROMPTS = {
        "cot": {
            "vision": "Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
            "standard": "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
        },
        "direct": {
            "vision": "Answer with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
            "standard": "Answer with the option letter from the given choices directly."
        }
    }
    
    def __init__(self):
        pass
    
    def replace_image_tokens(self, input_string: str) -> Tuple[str, List[int]]:
        """Replace <image i> tokens and record their original order.
        
        This function handles the shuffled image tokens in MMMU PRO Standard (10 options).
        
        Args:
            input_string: String containing <image i> tokens
            
        Returns:
            Tuple of (processed_string, image_order_list)
            - processed_string: String with tokens replaced
            - image_order_list: List of original image indices in order of appearance
        """
        # Find all <image i> tokens
        pattern = r'<image (\d+)>'
        matches = re.findall(pattern, input_string)
        
        # Record the order of image indices
        image_order = [int(match) for match in matches]
        
        # Replace tokens with placeholder for images
        processed_string = re.sub(pattern, '[IMAGE]', input_string)
        
        return processed_string, image_order
    
    def process_options_with_images(self, options: List[str], images: List[Image.Image]) -> Tuple[str, List[Image.Image]]:
        """Process options that may contain image tokens and reorder images accordingly.
        
        Args:
            options: List of option strings (may contain <image i> tokens)
            images: List of PIL Images (ordered by image_1, image_2, etc.)
            
        Returns:
            Tuple of (formatted_options_text, reordered_images)
        """
        if not options:
            return "", images
        
        # Create image mapping (1-indexed to 0-indexed)
        image_map = {i+1: img for i, img in enumerate(images)}
        
        formatted_options = []
        used_images = []
        
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            
            # Check if option contains image token
            if '<image' in option:
                processed_option, image_order = self.replace_image_tokens(option)
                
                # Add images in the order they appear in this option
                for img_idx in image_order:
                    if img_idx in image_map:
                        used_images.append(image_map[img_idx])
                
                formatted_options.append(f"{letter}. {processed_option}")
            else:
                formatted_options.append(f"{letter}. {option}")
        
        # If no images were used in options, return original images
        if not used_images:
            used_images = images
        
        options_text = "\n".join(formatted_options)
        return options_text, used_images
    
    def build_prompt(self, sample: Dict[str, Any], mode: str, setting: str) -> Tuple[str, List[Image.Image]]:
        """Build prompt for MMMU PRO inference using official prompts.
        
        Args:
            sample: Processed sample from dataset
            mode: Inference mode ('cot' or 'direct')
            setting: Dataset setting
            
        Returns:
            Tuple of (prompt_text, ordered_images)
        """
        if setting == "vision":
            # Vision mode: use official vision prompt (question embedded in image)
            prompt_text = self.OFFICIAL_PROMPTS[mode]["vision"]
            return prompt_text, sample['images']
        
        else:
            # Standard modes: handle question and options with image token processing
            question = sample['question']
            options = sample.get('options', [])
            images = sample.get('images', [])
            
            # Process options and handle image token reordering
            options_text, reordered_images = self.process_options_with_images(options, images)
            
            # For standard modes, we need to present the question and options first
            # then add the official prompt instruction
            if options_text:
                # Multi-choice question with options
                full_prompt = f"{question}\n\n{options_text}\n\n{self.OFFICIAL_PROMPTS[mode]['standard']}"
            else:
                # Question without explicit options
                full_prompt = f"{question}\n\n{self.OFFICIAL_PROMPTS[mode]['standard']}"
            
            return full_prompt, reordered_images


def extract_answer_from_response(response: str, setting: str, options: List[str] = None) -> str:
    """Extract answer from model response using official MMMU PRO logic.
    
    Official format: "Answer: $LETTER" (without quotes) where LETTER is one of options.
    
    Args:
        response: Raw model response
        setting: Dataset setting
        options: List of options (for standard modes)
        
    Returns:
        Extracted answer
    """
    import re
    
    response = response.strip()
    
    # First, look for the official format: "Answer: X"
    official_pattern = r'Answer:\s*([A-J])'
    matches = re.findall(official_pattern, response.upper(), re.IGNORECASE)
    if matches:
        answer = matches[-1].upper()  # Take the last occurrence
        # Validate answer is within valid range
        if options and ord(answer) - ord('A') < len(options):
            return answer
        elif not options:  # No options to validate against (vision mode)
            return answer
    
    # If no official format found, try other patterns
    patterns = [
        r'(?:final answer|the answer|my answer)(?:\s*is)?\s*:?\s*([A-J])',
        r'(?:select|choose|pick)\s*(?:option\s*)?([A-J])',
        r'(?:^|\n)\s*([A-J])\s*[.)]?\s*$',  # Letter at start of line
        r'\b([A-J])\b(?=\s*$)',  # Letter at end of response
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response.upper(), re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1].upper()
            # Validate answer is within valid range
            if options and ord(answer) - ord('A') < len(options):
                return answer
            elif not options:
                return answer
    
    # Last resort: find any single letter in the response
    letters = re.findall(r'\b([A-J])\b', response.upper())
    if letters:
        answer = letters[-1]
        if options and ord(answer) - ord('A') < len(options):
            return answer
        elif not options:
            return answer
    
    # If no letter found, return the full response for manual inspection
    return response


# Test the image token processing
if __name__ == "__main__":
    manager = MMMUProPromptManager()
    
    # Test image token replacement
    test_options = [
        "<image 2>",
        "<image 1>", 
        "<image 4>",
        "<image 3>"
    ]
    
    print("Testing image token processing:")
    for i, option in enumerate(test_options):
        processed, order = manager.replace_image_tokens(option)
        print(f"Option {i}: '{option}' -> '{processed}', order: {order}")
