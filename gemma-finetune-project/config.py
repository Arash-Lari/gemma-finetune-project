"""
Configuration file for Gemma fine-tuning project.
Modify these settings to customize your training setup.
"""

import os
from pathlib import Path

# Model Configuration
MODEL_CONFIG = {
    # Path to your Gemma-3-27B-it-Abliterated model
    "model_path": "~/models/mlabonne/gemma-3-27B-it-abliterated-GGUF/gemma-3-27B-it-abliterated.q4_k_m.gguf",
    
    # Path to vision model for PDF processing (optional)
    "vision_model_path": "~/models/gemma-3-vision-latex.Q4_K_S.gguf",
    
    # LM Studio API URL
    "vision_api_url": "http://localhost:1234/v1"
}

# Data Configuration
DATA_CONFIG = {
    # Path to training data directory
    "data_dir": "~/Desktop/trainingdata/",
    
    # Supported file extensions
    "supported_extensions": {'.pdf', '.xlsx', '.txt', '.csv'},
    
    # Maximum number of entries to process (None for all)
    "max_entries": None,
    
    # Number of previous entries to use as context
    "conversation_window": 5
}

# Training Configuration
TRAINING_CONFIG = {
    # LoRA parameters
    "lora_rank": 16,
    "lora_alpha": 32.0,
    "lora_dropout": 0.1,
    
    # Training parameters
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "max_seq_length": 2048,
    
    # Optimization
    "gradient_accumulation": 4,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    
    # LoRA target modules
    "target_modules": [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}

# Output Configuration
OUTPUT_CONFIG = {
    # Output directory
    "output_dir": Path("output"),
    
    # File names
    "parsed_data_file": "parsed_data.json",
    "train_data_file": "train.jsonl",
    "val_data_file": "train_val.jsonl",
    "results_file": "finetune_results.json",
    "log_file": "finetune.log"
}

# Vision Processing Configuration
VISION_CONFIG = {
    # Vision prompt for PDF extraction
    "pdf_extraction_prompt": """Analyze this phone chat screenshot and extract all messages in the following format:
    - Sender name
    - Timestamp (if visible)
    - Message content
    
    If this is a chat interface, identify each message bubble and extract the sender and content.
    If timestamps are visible, include them.
    Format the output as structured text that can be parsed.""",
    
    # Vision API parameters
    "max_tokens": 2000,
    "temperature": 0.1,
    "timeout": 60
}

# Dataset Preparation Configuration
DATASET_CONFIG = {
    # Gemma instruction format tokens
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "start_of_turn": "<start_of_turn>",
    "end_of_turn": "<end_of_turn>",
    
    # Validation split ratio
    "validation_ratio": 0.1,
    
    # Prompt types for instruction generation
    "prompt_types": [
        {
            "name": "emulation",
            "instruction_template": "Emulate {speaker}'s communication style based on this message: {content}",
            "output_template": "Here's how {speaker} would respond: {content}"
        },
        {
            "name": "analysis",
            "instruction_template": "Analyze this message from {speaker}: {content}",
            "output_template": "Analysis: This message shows {speaker}'s communication style and intent."
        },
        {
            "name": "conversation",
            "instruction_template": "Continue the conversation as {speaker}: {context}{speaker}: {content}",
            "output_template": "{speaker}: {content}"
        },
        {
            "name": "summary",
            "instruction_template": "Summarize this message from {speaker}: {content}",
            "output_template": "Summary: {summary}"
        }
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": True,
    "console_handler": True
}

# System Configuration
SYSTEM_CONFIG = {
    # Memory management
    "chunk_size": 1000,
    "max_memory_usage": 0.8,  # 80% of available RAM
    
    # Processing
    "num_workers": 4,
    "timeout": 300,  # 5 minutes
    
    # Error handling
    "max_retries": 3,
    "retry_delay": 1  # seconds
}

def get_config():
    """Get the complete configuration dictionary."""
    return {
        "model": MODEL_CONFIG,
        "data": DATA_CONFIG,
        "training": TRAINING_CONFIG,
        "output": OUTPUT_CONFIG,
        "vision": VISION_CONFIG,
        "dataset": DATASET_CONFIG,
        "logging": LOGGING_CONFIG,
        "system": SYSTEM_CONFIG
    }

def validate_config():
    """Validate the configuration and return any issues."""
    issues = []
    
    # Check model paths
    model_path = os.path.expanduser(MODEL_CONFIG["model_path"])
    if not os.path.exists(model_path):
        issues.append(f"Model not found: {model_path}")
    
    # Check data directory
    data_dir = os.path.expanduser(DATA_CONFIG["data_dir"])
    if not os.path.exists(data_dir):
        issues.append(f"Data directory not found: {data_dir}")
    
    # Check output directory
    output_dir = OUTPUT_CONFIG["output_dir"]
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    return issues

if __name__ == "__main__":
    """Test the configuration."""
    issues = validate_config()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!") 