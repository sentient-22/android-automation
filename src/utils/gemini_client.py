"""
Gemini API client for interacting with Google's Gemini models.
"""
import os
import logging
from typing import Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini models."""
    
    def __init__(self, model_name: str = "gemini-pro", api_key: str = None):
        """Initialize the Gemini client.
        
        Args:
            model_name: Name of the Gemini model to use (default: "gemini-pro")
            api_key: Optional API key. If not provided, will try to get from GEMINI_API_KEY env var
        """
        load_dotenv()
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable "
                "or pass it directly to the GeminiClient constructor."
            )
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini client with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, **generation_config) -> str:
        """Generate text using the Gemini model.
        
        Args:
            prompt: The prompt to generate text from
            **generation_config: Additional generation parameters
                - max_tokens: Maximum number of tokens to generate
                - temperature: Controls randomness (0.0 to 1.0)
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - generation_config: Dictionary of generation parameters (alternative to individual args)
                
        Returns:
            Generated text as a string
        """
        try:
            # Handle both direct parameters and generation_config dictionary
            config = generation_config.pop('generation_config', {})
            if 'max_tokens' in generation_config:
                config['max_output_tokens'] = generation_config.pop('max_tokens')
            
            # Add remaining generation parameters to config
            for key in ['temperature', 'top_p', 'top_k']:
                if key in generation_config:
                    config[key] = generation_config.pop(key)
            
            # Generate the response
            response = self.model.generate_content(
                prompt,
                generation_config=config if config else None,
                **generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_multimodal(
        self, 
        prompt: str, 
        image_path: str, 
        **generation_config
    ) -> str:
        """Generate text using both text and image inputs.
        
        Args:
            prompt: The text prompt
            image_path: Path to the image file
            **generation_config: Additional generation parameters
            
        Returns:
            Generated text as a string
        """
        try:
            # Load the image
            import base64
            from PIL import Image
            import io
            
            # Open and encode the image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to bytes
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_byte = buffered.getvalue()
            
            # Create the parts
            image_part = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte).decode("utf-8")
            }
            
            # Generate content
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content(
                contents=[prompt, image_part],
                **generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error in multimodal generation: {str(e)}")
            raise
    
    def get_embeddings(self, text: str) -> list[float]:
        """Get embeddings for the input text.
        
        Args:
            text: Input text to get embeddings for
            
        Returns:
            List of floats representing the embedding
        """
        try:
            # Use the embedding model
            embedding_model = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return embedding_model['embedding']
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize the client
    client = GeminiClient()
    
    # Example text generation
    response = client.generate_text("Tell me a joke about AI")
    print(f"Response: {response}")
    
    # Example multimodal generation (uncomment to use)
    # response = client.generate_multimodal(
    #     "What's in this image?",
    #     "path/to/your/image.jpg"
    # )
    # print(f"Multimodal response: {response}")
