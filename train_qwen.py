import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Argument Classes
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen-VL-Chat",
        metadata={"help": "Model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Enable trust_remote_code for loading models"}
    )

@dataclass
class DataTrainingArguments:
    dataset_path: str = field(
        default="data/vlm_dataset/metadata.jsonl",
        metadata={"help": "Path to the training data"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing images"}
    )

@dataclass
class LoraArguments:
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )

@dataclass
class QuantizationArguments:
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit precision (may not work on all systems)"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit"}
    )

def load_model_and_processor(model_args, training_args, quant_args):
    """Load Qwen-VL model and processor with workaround for Resampler issue."""
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch
        
        # Configure device and compute settings
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        
        # Workaround for Resampler issue
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
            original_init = Qwen2ForCausalLM._initialize_weights
            
            def patched_init(module):
                if hasattr(module, '_initialize_weights'):
                    return original_init(module)
                return None
            
            # Apply patch
            Qwen2ForCausalLM._initialize_weights = patched_init
            patch_applied = True
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not apply Resampler patch: {e}")
            patch_applied = False
        
        # Load model with trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        
        # Restore original method if patch was applied
        if patch_applied:
            Qwen2ForCausalLM._initialize_weights = original_init
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded on {model.device} with {model.dtype}")
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def load_and_prepare_dataset(data_args, processor):
    """Load and prepare dataset for training."""
    try:
        # Load dataset
        dataset = load_dataset("json", data_files={"train": data_args.dataset_path})["train"]
        logger.info(f"Loaded {len(dataset)} examples from {data_args.dataset_path}")
        
        # Simple text processing
        def process_examples(examples):
            return processor(
                text=examples["text"],
                padding="max_length",
                max_length=data_args.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
        
        # Process dataset
        dataset = dataset.map(
            process_examples,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Add labels
        dataset = dataset.map(
            lambda x: {"labels": x["input_ids"].clone()},
            batched=True
        )
        
        # Split into train/eval (90/10)
        if len(dataset) > 10:
            dataset = dataset.train_test_split(test_size=0.1)
            return dataset["train"], dataset["test"]
        return dataset, None
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def setup_training(model, training_args, lora_args):
    """Configure model for training with optional LoRA."""
    try:
        if lora_args.use_lora:
            logger.info("Preparing model for LoRA training")
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            
            # Default target modules for Qwen
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
            
            # Configure LoRA
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_args.lora_rank,
                lora_alpha=lora_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model
        
    except Exception as e:
        logger.error(f"Error setting up training: {e}")
        raise

def train():
    """Main training function."""
    try:
        # Parse arguments
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments, LoraArguments, QuantizationArguments)
        )
        
        # Parse command line arguments
        model_args, data_args, training_args, lora_args, quant_args = parser.parse_args_into_dataclasses()
        
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        
        # Log arguments
        logger.info(f"Model: {model_args.model_name_or_path}")
        logger.info(f"Dataset: {data_args.dataset_path}")
        logger.info(f"Output dir: {training_args.output_dir}")
        
        # Set seed for reproducibility
        set_seed(training_args.seed)
        
        # Load model and processor
        model, processor = load_model_and_processor(model_args, training_args, quant_args)
        
        # Set up training (LoRA, etc.)
        model = setup_training(model, training_args, lora_args)
        
        # Load and prepare dataset
        train_dataset, eval_dataset = load_and_prepare_dataset(data_args, processor)
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=processor.tokenizer,
            mlm=False,
        )
        
        # Initialize training arguments
        training_args = TrainingArguments(
            output_dir=training_args.output_dir,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            learning_rate=training_args.learning_rate,
            num_train_epochs=training_args.num_train_epochs,
            max_steps=training_args.max_steps,
            lr_scheduler_type=training_args.lr_scheduler_type,
            warmup_steps=training_args.warmup_steps,
            logging_steps=training_args.logging_steps,
            save_steps=training_args.save_steps,
            save_total_limit=training_args.save_total_limit,
            eval_strategy=training_args.eval_strategy,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            tf32=training_args.tf32,
            gradient_checkpointing=training_args.gradient_checkpointing,
            report_to=["tensorboard"],
            load_best_model_at_end=False,  # Disable to avoid issues with model loading
            remove_unused_columns=False,   # Keep all columns for the processor
            ddp_find_unused_parameters=False,  # Disable to avoid warnings
            dataloader_num_workers=0,      # Set to 0 to avoid multiprocessing issues
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        if training_args.output_dir is not None:
            logger.info(f"Saving model to {training_args.output_dir}")
            trainer.save_model(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
            
        logger.info("Training completed successfully!")
        return trainer
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train()


def main():
    """Main function to run the training script."""
    try:
        # Parse command line arguments
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments, LoraArguments, QuantizationArguments)
        )
        model_args, data_args, training_args, lora_args, quant_args = parser.parse_args_into_dataclasses()
        
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        
        # Log arguments
        logger.info(f"Model arguments: {model_args}")
        logger.info(f"Data arguments: {data_args}")
        logger.info(f"Training arguments: {training_args}")
        logger.info(f"LoRA arguments: {lora_args}")
        logger.info(f"Quantization arguments: {quant_args}")
        
        # Set seed for reproducibility
        set_seed(training_args.seed)
        
        # Load model and processor
        model, processor = load_model_and_processor(model_args, training_args, quant_args)
        
        # Set up training (LoRA, etc.)
        model = setup_training(model, training_args, lora_args)
        
        # Load and prepare dataset
        train_dataset, eval_dataset = load_and_prepare_dataset(data_args, training_args, processor)
        
        # Train the model
        train(model, processor, train_dataset, eval_dataset, training_args, lora_args)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
