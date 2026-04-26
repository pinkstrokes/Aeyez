"""
MIA-Bench Model Utils

This module provides a bridge between MIA-Bench requirements and our existing LLM infrastructure.
It reuses the existing app/llm.py while providing MIA-specific interfaces.
Following the pattern established in MMMU PRO model_utils.py.

Supports:
- Qwen2.5-VL-3B/7B/32B via DashScope API and vLLM
- GPT-4o-mini via OpenAI API
- Multi-agent flow preparation for future extension
"""

import os
import sys
import json
import base64
import requests
import io
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import asyncio

# Add parent directories to path for importing from app
current_dir = os.path.dirname(__file__)
multi_agent_dir = os.path.join(current_dir, '../../')
sys.path.insert(0, multi_agent_dir)

# Import handling similar to MMMU PRO
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    print("⚠️  PIL not available. Image processing may not work.")

try:
    from app.llm import LLM
    from app.schema import Message
    from app.config import config
    LLM_AVAILABLE = True
    print("✓ Successfully imported LLM system")
except ImportError as e:
    print(f"Warning: Could not import app.llm or app.config: {e}")
    LLM = None
    Message = None
    config = None
    LLM_AVAILABLE = False

class MIAModelWrapper:
    """
    Wrapper class that adapts our existing LLM infrastructure for MIA-Bench.
    
    This class:
    1. Uses our existing llm.py for all model interactions
    2. Handles MIA-specific image loading and processing
    3. Provides both single-model and flow-ready interfaces
    4. Maintains compatibility with official MIA evaluation format
    """
    
    def __init__(self, model_config: str = "mia_gpt4o_mini"):
        """
        Initialize the MIA model wrapper.
        
        Args:
            model_config: Configuration name from config.toml
                         Supported: mia_qwen2_5_vl_3b, mia_qwen2_5_vl_7b, mia_qwen2_5_vl_32b, mia_gpt4o_mini
        """
        if not LLM_AVAILABLE:
            raise ImportError("LLM system not available. Cannot create MIAModelWrapper.")
            
        self.model_config = model_config
        self.llm = LLM(config_name=model_config)
        
        # Store model info for logging and debugging
        self.model_name = self.llm.model
        self.api_type = self.llm.api_type
        
        print(f"🤖 Initialized MIA model wrapper:")
        print(f"   Config: {model_config}")
        print(f"   Model: {self.model_name}")
        print(f"   API Type: {self.api_type}")
    
    def load_image_from_url_or_path(self, image_url: str) -> str:
        """
        Load image from URL or local path and convert to base64 for our LLM interface.
        
        Args:
            image_url: URL of the image or local file path
            
        Returns:
            Base64 encoded image string
        """
        try:
            if not PIL_AVAILABLE:
                raise ImportError("PIL is required for image processing")
            
            # Check if it's a local file path or URL
            if image_url.startswith(('http://', 'https://')):
                # Load from URL (original behavior)
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            else:
                # Load from local path
                # Support both absolute and relative paths
                if not os.path.isabs(image_url):
                    # If relative path, make it relative to the data directory
                    data_dir = os.path.join(current_dir, 'data')
                    image_url = os.path.join(data_dir, image_url)
                
                if not os.path.exists(image_url):
                    raise FileNotFoundError(f"Image file not found: {image_url}")
                
                image = Image.open(image_url)
            
            # Convert to RGB if needed (for JPEG compatibility)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return base64_image
            
        except Exception as e:
            print(f"❌ Error loading image from {image_url}: {e}")
            raise
    
    async def generate_response(self, instruction: str, image_url: str) -> str:
        """
        Generate response for a single MIA sample.
        
        Args:
            instruction: The instruction/prompt for the model
            image_url: URL of the image to process
            
        Returns:
            Generated response text
        """
        try:
            # Load and encode image (supports both URL and local path)
            base64_image = self.load_image_from_url_or_path(image_url)
            
            # Create message with image using our existing Message class
            user_message = Message.user_message(
                content=instruction,
                base64_image=base64_image
            )
            
            # Use our existing LLM interface
            response = await self.llm.ask_with_images(
                messages=[user_message],
                images=[],  # Image is already in the message
                stream=False,
                temperature=0.0  # Deterministic for evaluation
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "error"
    
    def format_for_mia_evaluation(self, sample: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Format the response in the official MIA evaluation format.

        Args:
            sample: Original MIA sample with image URL and instruction
            response: Generated model response

        Returns:
            Dictionary in MIA evaluation format
        """
        # Get token usage from model
        token_usage = self.llm.token_tracker.get_usage_summary() if hasattr(self.llm, 'token_tracker') else {}

        # Use image URL as question_id (following official example format)
        return {
            "question_id": sample["image"],
            "prompt": sample["instruction"],
            "text": response,
            "token_usage": token_usage
        }


class MIAFlowWrapper:
    """
    Future-ready wrapper for multi-agent flow integration.
    
    This class provides the interface for integrating MIA-Bench with
    multi-agent flows while maintaining single-model compatibility.
    """
    
    def __init__(self, translator_config: str = "qwen2_5_vl_3b", 
                 reasoning_config: str = "text_only_reasoning"):
        """
        Initialize flow wrapper for future multi-agent integration.
        
        Args:
            translator_config: Vision model config for image understanding
            reasoning_config: Text model config for complex reasoning
        """
        self.translator_config = translator_config
        self.reasoning_config = reasoning_config
        
        # For now, use single model - can be extended for flow later
        self.model_wrapper = MIAModelWrapper(translator_config)
        
        print(f"🔄 Initialized MIA flow wrapper (single-model mode)")
        print(f"   Vision Model: {translator_config}")
        print(f"   Future Reasoning Model: {reasoning_config}")
    
    async def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single MIA sample (flow-ready interface).
        
        This method provides a unified interface that can be extended
        for multi-agent flows in the future.
        
        Args:
            sample: MIA sample dictionary
            
        Returns:
            Formatted result for evaluation
        """
        # For now, use single model
        response = await self.model_wrapper.generate_response(
            sample["instruction"], 
            sample["image"]
        )
        
        return self.model_wrapper.format_for_mia_evaluation(sample, response)
    
    # Future extension point for multi-agent flows
    async def process_with_flow(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Future method for multi-agent flow processing.
        
        This will integrate with TranslatorAgent + TextOnlyReasoningAgent
        similar to MMMU flow evaluation.
        """
        # TODO: Implement multi-agent flow integration
        # This would follow the pattern from run_mmmu_flow.py
        raise NotImplementedError("Multi-agent flow not yet implemented")


def create_model_wrapper(model_config: str) -> MIAModelWrapper:
    """
    Factory function to create MIA model wrapper.
    
    Args:
        model_config: Model configuration name
        
    Returns:
        Configured MIAModelWrapper instance
    """
    return MIAModelWrapper(model_config)


def create_flow_wrapper(translator_config: str = "qwen2_5_vl_3b", 
                       reasoning_config: str = "text_only_reasoning") -> MIAFlowWrapper:
    """
    Factory function to create MIA flow wrapper.
    
    Args:
        translator_config: Vision model configuration
        reasoning_config: Reasoning model configuration
        
    Returns:
        Configured MIAFlowWrapper instance
    """
    return MIAFlowWrapper(translator_config, reasoning_config)


# Supported model configurations for MIA-Bench
SUPPORTED_MODELS = {
    # MIA-specific configurations (optimized for evaluation)
    "mia_qwen2_5_vl_3b": "Qwen2.5-VL-3B-Instruct (MIA optimized, vLLM)",
    "mia_qwen2_5_vl_7b": "Qwen2.5-VL-7B-Instruct (MIA optimized, vLLM)", 
    "mia_qwen2_5_vl_32b": "Qwen2.5-VL-32B-Instruct (MIA optimized, DashScope)",
    "mia_gpt4o_mini": "GPT-4o-mini (MIA optimized, OpenAI API)",
    
    # 直接使用现有配置 (推荐)
    "qwen2_5_vl_3b": "Qwen2.5-VL-3B-Instruct (现有 vLLM 配置)",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B-Instruct (现有 vLLM 配置)",
    "qwen2_5_vl_32b": "Qwen2.5-VL-32B-Instruct (现有 DashScope 配置)",
    "gpt4o_mini": "GPT-4o-mini (现有 OpenAI 配置)",
    "translator": "Qwen2.5-VL-3B via DashScope (现有配置)",
    "qwen2_5_vl_7b_dashscope": "Qwen2.5-VL-7B via DashScope (现有配置)",
    
    # 新增 DashScope API 支持 3B 和 7B
    "mia_qwen2_5_vl_3b_dashscope": "Qwen2.5-VL-3B via DashScope API (新增)",
    "mia_qwen2_5_vl_7b_dashscope": "Qwen2.5-VL-7B via DashScope API (新增)",
}


def list_supported_models():
    """Print all supported model configurations."""
    print("🤖 Supported MIA model configurations:")
    for config, description in SUPPORTED_MODELS.items():
        print(f"   {config}: {description}")


if __name__ == "__main__":
    # Test the model wrapper
    import asyncio
    
    async def test_wrapper():
        print("Testing MIA Model Wrapper...")
        
        # Test with a simple configuration
        wrapper = create_model_wrapper("gpt4o_mini")  # Use GPT-4o-mini for testing
        
        # Test sample (using the example from official data)
        test_sample = {
            "image": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "instruction": "Explain the activity taking place in the image using exactly two sentences, including one metaphor."
        }
        
        try:
            result = await wrapper.process_sample(test_sample)
            print("✅ Test successful!")
            print(f"Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Run test if executed directly
    asyncio.run(test_wrapper())
