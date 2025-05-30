# Fine-tuning Qwen 1.5 for Android UI Automation

This guide explains how to fine-tune Qwen 1.5 on your Android UI automation dataset.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- At least 16GB of GPU RAM (for 7B model)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-train.txt
   ```

3. Install PyTorch with CUDA support (if using GPU):
   ```bash
   # For CUDA 11.8
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Training

1. **Prepare your dataset**:
   - Ensure your dataset is in the format of `metadata.jsonl` in the `data/vlm_dataset/` directory
   - Each example should have the structure shown in the existing dataset

2. **Start training**:
   ```bash
   python train_qwen.py
   ```

   For multi-GPU training:
   ```bash
   torchrun --nproc_per_node=4 train_qwen.py
   ```

## Configuration

You can modify the training parameters in the `TrainingConfig` class in `train_qwen.py`:

- `model_name`: Base model to fine-tune (default: "Qwen/Qwen1.5-7B")
- `batch_size`: Batch size per device (reduce if OOM)
- `learning_rate`: Learning rate (default: 2e-4)
- `num_train_epochs`: Number of training epochs
- `use_4bit`: Enable 4-bit quantization (recommended)

## Monitoring

Training logs and checkpoints will be saved to the `models/qwen_finetuned` directory by default. You can monitor training with TensorBoard:

```bash
tensorboard --logdir=models/qwen_finetuned/runs
```

## Using the Fine-tuned Model

After training, load your model with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/qwen_finetuned/final_model"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

## Tips

- Start with a smaller learning rate (2e-5 to 5e-5) for better stability
- Use gradient accumulation for larger effective batch sizes
- Monitor loss on a validation set
- Consider using DeepSpeed for more efficient training
- For production, you may want to convert the model to ONNX or TensorRT for better performance
