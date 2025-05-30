# LLM Android Automation POC

A proof-of-concept for LLM-powered Android automation using Appium and Gemini, with built-in dataset generation for Vision-Language Model (VLM) training and Qwen fine-tuning capabilities.

A proof-of-concept for LLM-powered Android automation using Appium and Gemini, with built-in dataset generation for Vision-Language Model (VLM) training.

## ✨ Features

- **LLM-Powered Automation**: Uses Gemini to understand and interact with Android UI
- **Multi-modal Understanding**: Processes both visual (screenshots) and structural (UI hierarchy) information
- **Dataset Generation**: Automatically collects training data in a format suitable for VLM fine-tuning
- **Qwen Fine-tuning**: Tools for fine-tuning Qwen models on collected datasets
- **Extensible Architecture**: Easy to add new actions and integrate with different LLM providers
- **Comprehensive Logging**: Detailed logs for debugging and analysis

## 🏗️ Project Structure

```
llm-android-automation/
├── .gitignore               # Git ignore rules for Python/ML projects
├── cleanup.sh               # Script to clean temporary files
├── config/                  # Configuration files
│   ├── appium_config.yaml   # Appium device and server settings
│   └── gemini_config.yaml   # Gemini API configuration
├── data/                    # Data directory
│   ├── screenshots/        # Screenshots captured during automation
│   └── vlm_dataset/        # Training data for VLM fine-tuning
│       ├── images/         # Screenshots for training
│       └── metadata.jsonl  # Training metadata
├── logs/                    # Log files
├── output/                  # Model outputs and checkpoints
├── src/                     # Source code
│   ├── agents/              # Agent implementations
│   ├── data/               # Data handling code
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── evaluate_agent.py        # Evaluation script
├── main.py                 # Main entry point
├── requirements.txt        # Main requirements
├── requirements-eval.txt   # Evaluation requirements
├── requirements-train.txt  # Training requirements
└── train_qwen.py          # Qwen fine-tuning script

## 🧹 Cleanup

To clean up temporary files and caches:

```bash
# Make the cleanup script executable
chmod +x cleanup.sh

# Run the cleanup script
./cleanup.sh
```

This will remove:
- Python cache files (`__pycache__`, `*.pyc`, etc.)
- Build and distribution directories
- IDE-specific files
- Log files
- Temporary files
- Jupyter notebook checkpoints

## 🗄️ Git Ignore

The project includes a `.gitignore` file that excludes:
- Python cache and compiled files
- Virtual environments
- IDE-specific files
- Data and model files
- Logs and outputs
- Environment variables
- Temporary files
## 📦 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm-android-automation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. Configure your environment:
   - Update `config/appium_config.yaml` with your device details
   - Add your Gemini API key to `config/gemini_config.yaml`

2. Run the agent:
   ```bash
   python main.py --task "Your task description" --max-steps 10
   ```

## 🤖 Training

To fine-tune the Qwen model on your dataset:

1. Install training requirements:
   ```bash
   pip install -r requirements-train.txt
   ```

2. Run the training script:
   ```bash
   python train_qwen.py \
     --model_name_or_path Qwen/Qwen-VL-Chat \
     --dataset_path data/vlm_dataset/metadata.jsonl \
     --output_dir ./output
   ```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - The Vision-Language Model used
- [Appium](https://appium.io/) - Mobile test automation
- [Gemini](https://ai.google.dev/) - For LLM capabilities
## 📊 Evaluation

To evaluate the agent's performance:

```bash
python evaluate_agent.py \
  --model_path ./output/checkpoint-1000 \
  --test_data data/vlm_dataset/test.jsonl \
  --output_dir ./results
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - The Vision-Language Model used
- [Appium](https://appium.io/) - Mobile test automation
- [Gemini](https://ai.google.dev/) - For LLM capabilities
## 📊 Evaluation

To evaluate the agent's performance:

```bash
python evaluate_agent.py \
  --model_path ./output/checkpoint-1000 \
  --test_data data/vlm_dataset/test.jsonl \
  --output_dir ./results
```

Metrics tracked:
- Task success rate
- Steps to completion
- Action accuracy
- Response time

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - The Vision-Language Model used
- [Appium](https://appium.io/) - Mobile test automation
- [Gemini](https://ai.google.dev/) - For LLM capabilities
## 📊 Evaluation

To evaluate the agent's performance:

```bash
python evaluate_agent.py \
  --model_path ./output/checkpoint-1000 \
  --test_data data/vlm_dataset/test.jsonl \
  --output_dir ./results
```

Metrics tracked:
- Task success rate
- Steps to completion
- Action accuracy
- Response time

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - The Vision-Language Model used
- [Appium](https://appium.io/) - Mobile test automation
- [Gemini](https://ai.google.dev/) - For LLM capabilities
├── model_cache/            # Downloaded model cache
├── output/                 # Training outputs
│   ├── models/            # Fine-tuned models
│   └── logs/              # Training logs
├── tests/                  # Test files
├── train_qwen.py           # Qwen fine-tuning script
├── requirements-train.txt  # Training dependencies
├── main.py                # Main entry point
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Fine-tuning Qwen

1. Install training dependencies:
   ```bash
   pip install -r requirements-train.txt
   ```

2. Run fine-tuning (CPU mode):
   ```bash
   python train_qwen.py \
     --model_name_or_path Qwen/Qwen1.5-0.5B-Chat \
     --output_dir ./output/models/qwen_finetuned \
     --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 4 \
     --learning_rate 1e-5 \
     --num_train_epochs 1
   ```

3. For private/gated models, use authentication:
   ```bash
   # Using environment variable
   export HF_TOKEN=your_token_here
   python train_qwen.py
   
   # Or via command line
   python train_qwen.py --hf_token your_token_here
   
   # Or use interactive prompt
   python train_qwen.py --use_auth_token
   ```

4. Monitor training:
   ```bash
   tail -f output/logs/automation.log
   ```

### Advanced Options

- `--model_name_or_path`: Model identifier from Hugging Face Hub or local path
- `--cache_dir`: Directory to cache downloaded models (default: ~/.cache/huggingface/hub)
- `--lora_rank`: LoRA rank (default: 4)
- `--lora_alpha`: LoRA alpha (default: 8)
- `--lora_dropout`: LoRA dropout (default: 0.05)
- `--no_cuda`: Force CPU training
- `--fp16/--bf16`: Enable mixed precision training (requires GPU)

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36.0+
- PEFT 0.7.0+
- Datasets 2.14.0+

### Prerequisites

- Python 3.8+
- Node.js and npm (for Appium)
- Android SDK and emulator/device
- Java Development Kit (JDK)
- For training: At least 16GB RAM (32GB recommended)
- For GPU training: CUDA-compatible GPU with at least 12GB VRAM

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-android-automation.git
   cd llm-android-automation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment:
   - Update `config/appium_config.yaml` with your device details
   - Add your Gemini API key to `config/gemini_config.yaml`

### Running the Agent

1. Start Appium server in a separate terminal:
   ```bash
   appium
   ```

2. Start your Android emulator or connect a physical device

3. Run the agent with a task:
   ```bash
   python main.py --task "Open the settings app" --max-steps 10 --log-level INFO
   ```

### Command Line Options

```
usage: main.py [-h] [--task TASK] [--max-steps MAX_STEPS] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--output-dir OUTPUT_DIR]

LLM Android Automation POC

options:
  -h, --help            show this help message and exit
  --task TASK           Task to perform (default: "Open the settings app")
  --max-steps MAX_STEPS
                        Maximum number of steps to perform (default: 10)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO)
  --output-dir OUTPUT_DIR
                        Output directory for logs and data (default: output)
```

## 📊 Dataset Generation

The agent automatically collects training data in the `data/vlm_dataset` directory. Each interaction includes:

- Screenshot of the current screen
- UI hierarchy (XML)
- Generated action with reasoning
- Task context
- Timestamp and metadata

### Exporting for Training

To export the collected data in a format suitable for VLM training:

```python
from src.data.dataset import VLMDataset

dataset = VLMDataset("data/vlm_dataset")
dataset.export_for_training("data/training_data.jsonl")
```

The exported format is compatible with popular VLM frameworks like Qwen 1.5 and LLaVA.

## 🛠️ Development

### Code Style

We use:
- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking

Run the following before committing:

```bash
black .
isort .
flake8
mypy .
```

### Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Appium](https://appium.io/) for mobile automation
- [Gemini](https://ai.google.dev/) for the LLM capabilities
- The open-source community for inspiration and tools
