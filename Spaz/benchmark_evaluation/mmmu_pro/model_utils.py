"""
Model utilities for MMMU PRO benchmark.
Provides unified interface for different model types using the existing LLM system.
"""

import os
import sys
import base64
import io
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import requests
import json

# Add parent directories to path for importing from app
current_dir = os.path.dirname(__file__)
multi_agent_dir = os.path.join(current_dir, '../../')
sys.path.insert(0, multi_agent_dir)

try:
    from app.llm import LLM
    from app.config import config
    LLM_AVAILABLE = True
    print("✓ Successfully imported LLM system")
except ImportError as e:
    print(f"Warning: Could not import app.llm or app.config: {e}")
    LLM = None
    config = None
    LLM_AVAILABLE = False


class ModelInterface:
    """Base class for model interfaces."""
    
    def __init__(self, model_name: str, config_name: str):
        self.model_name = model_name
        self.config_name = config_name
        
    async def generate_response(self, prompt: str, images: List[Image.Image] = None) -> str:
        """Generate response from model.
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            
        Returns:
            Generated response text
        """
        raise NotImplementedError


class UnifiedModelInterface(ModelInterface):
    """Unified interface using the existing LLM system."""
    
    def __init__(self, model_name: str, config_name: str):
        super().__init__(model_name, config_name)
        if not LLM_AVAILABLE:
            raise ImportError("LLM system not available. Cannot create UnifiedModelInterface.")
        
        # Create LLM instance with the specified config
        self.llm = LLM(config_name=config_name)
        
        # Expose token_tracker for compatibility with MMMU pattern
        self.token_tracker = self.llm.token_tracker
        
    def _convert_images_to_urls(self, images: List[Image.Image]) -> List[dict]:
        """Convert PIL Images to base64 data URLs for LLM system."""
        if not images:
            return []
            
        image_urls = []
        for img in images:
            # Resize image if too large to avoid DashScope 10MB limit
            max_size = (1024, 1024)  # Maximum dimensions
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img = img.copy()
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"Resized image from original size to {img.size}")
            
            # Convert PIL Image to base64 with compression
            buffer = io.BytesIO()
            # Use JPEG with quality=85 for better compression, PNG for images with transparency
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img.save(buffer, format='PNG', optimize=True)
                format_str = "png"
            else:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                format_str = "jpeg"
            
            img_data = buffer.getvalue()
            img_size_mb = len(img_data) / (1024 * 1024)
            
            # Check if still too large (DashScope limit is ~10MB)
            if img_size_mb > 8:  # Use 8MB as safe limit
                print(f"Warning: Image still large ({img_size_mb:.1f}MB), further compression needed")
                # Further reduce quality for JPEG
                if format_str == "jpeg":
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=60, optimize=True)
                    img_data = buffer.getvalue()
                    img_size_mb = len(img_data) / (1024 * 1024)
                    print(f"Compressed to {img_size_mb:.1f}MB")
            
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Create data URL format expected by LLM system
            image_urls.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{format_str};base64,{img_base64}"
                }
            })
        return image_urls
        
    async def generate_response(self, prompt: str, images: List[Image.Image] = None) -> str:
        """Generate response using the unified LLM system.
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            
        Returns:
            Generated response text
        """
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            if images and len(images) > 0:
                # Use ask_with_images for multimodal queries
                image_urls = self._convert_images_to_urls(images)
                response = await self.llm.ask_with_images(
                    messages=messages,
                    images=image_urls,
                    stream=False,
                    temperature=0.0  # Use zero temperature for maximum consistency
                )
            else:
                # Use regular ask for text-only queries
                response = await self.llm.ask(
                    messages=messages,
                    stream=False,
                    temperature=0.0
                )
                
            return response
            
        except Exception as e:
            print(f"Error in UnifiedModelInterface: {e}")
            raise


class UnifiedModelInterfaceNoCompression(ModelInterface):
    """Unified interface using the existing LLM system without image compression (for GPT models)."""
    
    def __init__(self, model_name: str, config_name: str):
        super().__init__(model_name, config_name)
        if not LLM_AVAILABLE:
            raise ImportError("LLM system not available. Cannot create UnifiedModelInterfaceNoCompression.")
        
        # Create LLM instance with the specified config
        self.llm = LLM(config_name=config_name)
        
        # Expose token_tracker for compatibility with MMMU pattern
        self.token_tracker = self.llm.token_tracker
        
    def _convert_images_to_urls(self, images: List[Image.Image]) -> List[dict]:
        """Convert PIL Images to base64 data URLs without compression (for GPT models)."""
        if not images:
            return []
            
        image_urls = []
        for img in images:
            # Convert PIL Image to base64 with moderate compression for GPT models
            buffer = io.BytesIO()
            
            # Apply moderate resizing for GPT models to avoid token limits
            max_size = (1536, 1536)  # Larger than Qwen but still manageable
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img = img.copy()
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"Resized image for GPT from original to {img.size}")
            
            # Use moderate quality to balance size and quality
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img.save(buffer, format='PNG', optimize=True)
                format_str = "png"
            else:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(buffer, format='JPEG', quality=90, optimize=True)  # Good quality but compressed
                format_str = "jpeg"
            
            img_data = buffer.getvalue()
            img_size_mb = len(img_data) / (1024 * 1024)
            print(f"Image size: {img_size_mb:.1f}MB (no compression)")
            
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Create data URL format expected by LLM system
            image_urls.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{format_str};base64,{img_base64}"
                }
            })
        return image_urls
        
    async def generate_response(self, prompt: str, images: List[Image.Image] = None) -> str:
        """Generate response using the unified LLM system without image compression."""
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            if images and len(images) > 0:
                # Use ask_with_images for multimodal queries
                image_urls = self._convert_images_to_urls(images)
                response = await self.llm.ask_with_images(
                    messages=messages,
                    images=image_urls,
                    stream=False,
                    temperature=0.0  # Use zero temperature for maximum consistency
                )
            else:
                # Use regular ask for text-only queries
                response = await self.llm.ask(
                    messages=messages,
                    stream=False,
                    temperature=0.0
                )
                
            return response
            
        except Exception as e:
            print(f"Error in UnifiedModelInterfaceNoCompression: {e}")
            raise


class QwenModelInterface(ModelInterface):
    """Interface for Qwen2.5-VL models via DashScope."""
    
    def __init__(self, model_name: str, config_name: str):
        super().__init__(model_name, config_name)
        
        # Check environment variables
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.base_url = os.getenv('DASHSCOPE_API_BASE')
        
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        if not self.base_url:
            raise ValueError("DASHSCOPE_API_BASE environment variable not set")
            
        # Initialize LLM if available
        if LLM is not None:
            try:
                self.llm = LLM.create(config_name)
                print(f"✓ Initialized Qwen model: {config_name}")
            except Exception as e:
                print(f"✗ Failed to initialize LLM: {e}")
                self.llm = None
        else:
            self.llm = None
            
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    async def generate_response(self, prompt: str, images: List[Image.Image] = None) -> str:
        """Generate response using Qwen model.
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            
        Returns:
            Generated response text
        """
        if self.llm is not None:
            # Use app.llm interface
            try:
                # Prepare messages
                messages = []
                
                if images:
                    # For multimodal input
                    content = []
                    content.append({"type": "text", "text": prompt})
                    
                    for img in images:
                        img_b64 = self.encode_image(img)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        })
                    
                    messages.append({"role": "user", "content": content})
                else:
                    # Text-only input
                    messages.append({"role": "user", "content": prompt})
                
                # Generate response
                response = await self.llm.ask(messages)
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                print(f"Error using app.llm: {e}")
                # Fall back to direct API call
                
        # Direct API call to DashScope
        return await self._direct_api_call(prompt, images)
    
    async def _direct_api_call(self, prompt: str, images: List[Image.Image] = None) -> str:
        """Make direct API call to DashScope.
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            
        Returns:
            Generated response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare messages
        messages = []
        
        if images:
            content = []
            content.append({"type": "text", "text": prompt})
            
            for img in images:
                img_b64 = self.encode_image(img)
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.001,
            "max_tokens": 4096
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"DashScope API error: {e}")
            return f"Error: {str(e)}"


class GPTModelInterface(ModelInterface):
    """Interface for GPT models via OpenAI API."""
    
    def __init__(self, model_name: str, config_name: str):
        super().__init__(model_name, config_name)
        
        # Check environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        # Initialize LLM if available
        if LLM is not None:
            try:
                self.llm = LLM.create(config_name)
                print(f"✓ Initialized GPT model: {config_name}")
            except Exception as e:
                print(f"✗ Failed to initialize LLM: {e}")
                self.llm = None
        else:
            self.llm = None
    
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    async def generate_response(self, prompt: str, images: List[Image.Image] = None) -> str:
        """Generate response using GPT model.
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            
        Returns:
            Generated response text
        """
        if self.llm is not None:
            # Use app.llm interface
            try:
                messages = []
                
                if images:
                    content = []
                    content.append({"type": "text", "text": prompt})
                    
                    for img in images:
                        img_b64 = self.encode_image(img)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        })
                    
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": prompt})
                
                response = await self.llm.ask(messages)
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                print(f"Error using app.llm: {e}")
                
        # Direct OpenAI API call
        return await self._direct_api_call(prompt, images)
    
    async def _direct_api_call(self, prompt: str, images: List[Image.Image] = None) -> str:
        """Make direct API call to OpenAI.
        
        Args:
            prompt: Text prompt
            images: List of PIL Images
            
        Returns:
            Generated response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        
        if images:
            content = []
            content.append({"type": "text", "text": prompt})
            
            for img in images:
                img_b64 = self.encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4096
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"


def create_model_interface(model_name: str, config_name: str) -> ModelInterface:
    """Create appropriate model interface based on model name.
    
    Args:
        model_name: Name of the model
        config_name: Configuration name in config.toml
        
    Returns:
        Model interface instance
    """
    # Prefer unified interface if LLM system is available
    if LLM_AVAILABLE:
        print(f"✓ Using unified LLM interface for {model_name}")
        # For GPT models, use no compression version
        if 'gpt' in model_name.lower():
            return UnifiedModelInterfaceNoCompression(model_name, config_name)
        else:
            return UnifiedModelInterface(model_name, config_name)
    
    # Fallback to direct API interfaces
    print(f"Warning: Using direct API interface for {model_name}")
    if 'qwen' in model_name.lower():
        return QwenModelInterface(model_name, config_name)
    elif 'gpt' in model_name.lower():
        return GPTModelInterface(model_name, config_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# Model name mappings
MODEL_MAPPINGS = {
    # Qwen models
    "qwen2.5-vl-3b": "qwen2.5-vl-3b-instruct",
    "qwen2.5-vl-7b": "qwen2.5-vl-7b-instruct", 
    "qwen2.5-vl-32b": "qwen2.5-vl-32b-instruct",
    
    # GPT models
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt4o-mini": "gpt-4o-mini",
}

CONFIG_MAPPINGS = {
    # Map model names to config.toml section names
    "qwen2.5-vl-3b": "translator",  # Use DashScope 3B config
    "qwen2.5-vl-7b": "qwen2_5_vl_7b_dashscope",  # Use DashScope 7B config
    "qwen2.5-vl-32b": "qwen2_5_vl_32b",  # Use DashScope 32B config
    "gpt-4o-mini": "gpt4o_mini",
    "gpt4o-mini": "gpt4o_mini",
}
