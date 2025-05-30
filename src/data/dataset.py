"""
Dataset module for collecting and managing VLM (Vision-Language Model) training data.
This follows a format similar to Qwen 1.5 for instruction-following tasks.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import uuid

from ..utils.logger import logger

class VLMDataset:
    """Dataset manager for Vision-Language Model training data."""
    
    def __init__(self, dataset_dir: Union[str, Path] = "data/vlm_dataset"):
        """Initialize the dataset manager.
        
        Args:
            dataset_dir: Base directory for the dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.metadata_file = self.dataset_dir / "metadata.jsonl"
        self._setup_directories()
        
        # Load existing metadata if available
        self.metadata = self._load_metadata()
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load existing metadata from file."""
        if not self.metadata_file.exists():
            return []
            
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return []
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                for item in self.metadata:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add_sample(
        self,
        image_path: Union[str, Path],
        instruction: str,
        input_text: str = "",
        output_text: str = "",
        ui_hierarchy: Optional[Dict] = None,
        action: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a new sample to the dataset.
        
        Args:
            image_path: Path to the screenshot/image
            instruction: The instruction for this sample
            input_text: Input text (context)
            output_text: Expected output text
            ui_hierarchy: UI hierarchy data
            action: Action taken (if any)
            metadata: Additional metadata
            
        Returns:
            Sample ID
        """
        # Generate unique ID for this sample
        sample_id = str(uuid.uuid4())
        
        # Copy image to dataset directory
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image_ext = image_path.suffix
        new_image_name = f"{sample_id}{image_ext}"
        new_image_path = self.images_dir / new_image_name
        
        try:
            shutil.copy2(image_path, new_image_path)
        except Exception as e:
            logger.error(f"Error copying image {image_path}: {e}")
            raise
        
        # Create sample data
        sample = {
            "id": sample_id,
            "image": str(new_image_path.relative_to(self.dataset_dir)),
            "conversations": [
                {
                    "from": "human",
                    "value": instruction
                },
                {
                    "from": "assistant",
                    "value": output_text
                }
            ],
            "input_text": input_text,
            "ui_hierarchy": ui_hierarchy or {},
            "action": action or {},
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to metadata and save
        self.metadata.append(sample)
        self._save_metadata()
        
        logger.info(f"Added sample {sample_id} to dataset")
        return sample_id
    
    def get_sample(self, sample_id: str) -> Optional[Dict]:
        """Get a sample by ID."""
        for sample in self.metadata:
            if sample.get('id') == sample_id:
                return sample
        return None
    
    def get_all_samples(self) -> List[Dict]:
        """Get all samples in the dataset."""
        return self.metadata
    
    def export_for_training(self, output_file: Union[str, Path]) -> None:
        """Export dataset in a format suitable for VLM training.
        
        This creates a JSONL file where each line is a JSON object with:
        - image: path to the image
        - conversations: list of conversation turns
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        training_data = []
        for sample in self.metadata:
            training_sample = {
                "id": sample["id"],
                "image": str(self.dataset_dir / sample["image"]),
                "conversations": sample["conversations"]
            }
            training_data.append(training_sample)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Exported {len(training_data)} samples to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            raise
