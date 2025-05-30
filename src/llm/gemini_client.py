import os
import base64
from typing import Dict, Any, Optional
import google.generativeai as genai
import yaml

class GeminiClient:
    def __init__(self, config_path: str = 'config/gemini_config.yaml'):
        self.config = self._load_config(config_path)
        self._initialize_client()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_client(self):
        """Initialize the Gemini client with API key"""
        api_key = self.config['gemini']['api_key']
        if not api_key or api_key == 'your-gemini-api-key':
            raise ValueError("Please set your Gemini API key in config/gemini_config.yaml")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config['gemini']['model'])
        self.generation_config = {
            'max_output_tokens': self.config['gemini']['max_tokens'],
            'temperature': self.config['gemini']['temperature'],
            'top_p': self.config['gemini']['top_p'],
            'top_k': self.config['gemini']['top_k'],
        }
    
    def generate_text(self, prompt: str, image_data: Optional[bytes] = None) -> str:
        """Generate text using the Gemini model"""
        try:
            if image_data:
                # For image+text input
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                response = self.model.generate_content(
                    contents=[
                        {
                            'parts': [
                                {'text': prompt},
                                {
                                    'inline_data': {
                                        'mime_type': 'image/png',
                                        'data': image_base64
                                    }
                                }
                            ]
                        }
                    ],
                    generation_config=self.generation_config
                )
            else:
                # For text-only input
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
            
            return response.text
        except Exception as e:
            return f"Error generating text: {str(e)}"
    
    def get_system_prompt(self) -> str:
        """Get the system prompt from config"""
        return self.config['prompt']['system_prompt']
