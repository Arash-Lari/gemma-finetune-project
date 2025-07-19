#!/usr/bin/env python3
"""
Gemma-3-27B-it-Abliterated Fine-tuning Script

This script fine-tunes the Gemma-3-27B-it-Abliterated model on personal chat logs
and journal data using MLX-LM with LoRA adapters.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate, LoRALinear
from mlx_lm.lora import LoRALinear, apply_lora_layers
from mlx_lm.utils import load_config, load_tokenizer
from mlx_lm.tuner import train

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/finetune.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GemmaFineTuner:
    def __init__(self, model_path: str, vision_model_path: Optional[str] = None):
        """
        Initialize the fine-tuner.
        
        Args:
            model_path: Path to the Gemma-3-27B-it-Abliterated model
            vision_model_path: Path to vision model for PDF processing (optional)
        """
        self.model_path = os.path.expanduser(model_path)
        self.vision_model_path = vision_model_path
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Verify model path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        logger.info(f"Initializing fine-tuner with model: {self.model_path}")
        
        # Check if model is abliterated version
        self._verify_abliterated_model()
        
    def _verify_abliterated_model(self):
        """Verify that the model is the abliterated version."""
        model_name = os.path.basename(self.model_path).lower()
        if "abliterated" not in model_name:
            logger.warning(f"Model name '{model_name}' doesn't contain 'abliterated'. "
                         f"Please ensure you're using the correct abliterated version.")
        else:
            logger.info("Confirmed: Using abliterated model version")
    
    def convert_gguf_to_mlx(self, gguf_path: str) -> str:
        """
        Convert GGUF model to MLX format if needed.
        
        Args:
            gguf_path: Path to GGUF model
            
        Returns:
            Path to MLX model
        """
        logger.info(f"Converting GGUF model to MLX: {gguf_path}")
        
        # Create output path
        mlx_path = gguf_path.replace('.gguf', '_mlx')
        if os.path.exists(mlx_path):
            logger.info(f"MLX model already exists at: {mlx_path}")
            return mlx_path
        
        try:
            # Use mlx_lm.convert to convert GGUF to MLX
            from mlx_lm.convert import convert
            
            convert(gguf_path, mlx_path)
            logger.info(f"Successfully converted to MLX: {mlx_path}")
            return mlx_path
            
        except Exception as e:
            logger.error(f"Error converting GGUF to MLX: {e}")
            raise
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        logger.info("Loading model and tokenizer...")
        
        try:
            # Load model configuration
            config = load_config(self.model_path)
            
            # Load tokenizer
            tokenizer = load_tokenizer(self.model_path)
            
            # Load model
            model = load(self.model_path)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model config: {config}")
            
            return model, tokenizer, config
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_lora_config(self, lora_rank: int = 16, lora_alpha: float = 32.0) -> Dict[str, Any]:
        """
        Prepare LoRA configuration.
        
        Args:
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha parameter
            
        Returns:
            LoRA configuration dictionary
        """
        return {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
    
    def train_lora(self, model, tokenizer, train_data_path: str, 
                   val_data_path: Optional[str] = None,
                   epochs: int = 3,
                   batch_size: int = 4,
                   learning_rate: float = 1e-5,
                   lora_rank: int = 16,
                   max_seq_length: int = 2048) -> str:
        """
        Train LoRA adapters on the model.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            train_data_path: Path to training data JSONL
            val_data_path: Path to validation data JSONL (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            lora_rank: LoRA rank
            max_seq_length: Maximum sequence length
            
        Returns:
            Path to saved LoRA adapters
        """
        logger.info("Starting LoRA training...")
        
        # Prepare LoRA config
        lora_config = self.prepare_lora_config(lora_rank=lora_rank)
        
        # Prepare training arguments
        train_args = {
            "model": model,
            "tokenizer": tokenizer,
            "train_data_path": train_data_path,
            "val_data_path": val_data_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "lora_config": lora_config,
            "output_dir": str(self.output_dir / "adapters"),
            "save_every": 100,  # Save every 100 steps
            "eval_every": 50,    # Evaluate every 50 steps
            "warmup_steps": 100,
            "gradient_accumulation": 4,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "lr_scheduler": "cosine",
            "save_best": True,
            "load_best": True
        }
        
        logger.info(f"Training configuration:")
        for key, value in train_args.items():
            if key not in ['model', 'tokenizer']:
                logger.info(f"  {key}: {value}")
        
        try:
            # Start training
            adapter_path = train(**train_args)
            
            logger.info(f"LoRA training completed. Adapters saved to: {adapter_path}")
            return adapter_path
            
        except Exception as e:
            logger.error(f"Error during LoRA training: {e}")
            raise
    
    def merge_lora_adapters(self, adapter_path: str) -> str:
        """
        Merge LoRA adapters with the base model.
        
        Args:
            adapter_path: Path to LoRA adapters
            
        Returns:
            Path to merged model
        """
        logger.info(f"Merging LoRA adapters from: {adapter_path}")
        
        try:
            from mlx_lm.fuse import fuse_lora
            
            # Load the base model
            model, tokenizer, config = self.load_model_and_tokenizer()
            
            # Merge LoRA adapters
            merged_model = fuse_lora(model, adapter_path)
            
            # Save merged model
            merged_path = str(self.output_dir / "merged_model")
            merged_model.save_weights(merged_path)
            
            # Save tokenizer and config
            tokenizer.save_pretrained(merged_path)
            with open(os.path.join(merged_path, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Merged model saved to: {merged_path}")
            return merged_path
            
        except Exception as e:
            logger.error(f"Error merging LoRA adapters: {e}")
            raise
    
    def convert_to_gguf(self, mlx_model_path: str) -> str:
        """
        Convert MLX model to GGUF format for LM Studio compatibility.
        
        Args:
            mlx_model_path: Path to MLX model
            
        Returns:
            Path to GGUF model
        """
        logger.info(f"Converting MLX model to GGUF: {mlx_model_path}")
        
        gguf_path = mlx_model_path.replace("_mlx", "_gguf.gguf")
        
        try:
            # Use llama.cpp to convert MLX to GGUF
            cmd = [
                "llama-cpp-python",  # or the appropriate converter
                "--mlx-model", mlx_model_path,
                "--output", gguf_path,
                "--outtype", "q4_k_m"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully converted to GGUF: {gguf_path}")
                return gguf_path
            else:
                logger.error(f"Error converting to GGUF: {result.stderr}")
                raise Exception(f"GGUF conversion failed: {result.stderr}")
                
        except FileNotFoundError:
            logger.warning("llama-cpp-python not found. Skipping GGUF conversion.")
            return mlx_model_path
        except Exception as e:
            logger.error(f"Error during GGUF conversion: {e}")
            return mlx_model_path
    
    def run_finetuning_pipeline(self, data_dir: str, 
                               epochs: int = 3,
                               batch_size: int = 4,
                               learning_rate: float = 1e-5,
                               lora_rank: int = 16,
                               max_entries: int = None,
                               convert_to_gguf: bool = False) -> Dict[str, str]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            data_dir: Path to training data directory
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            lora_rank: LoRA rank
            max_entries: Maximum number of entries to process
            convert_to_gguf: Whether to convert final model to GGUF
            
        Returns:
            Dictionary with paths to outputs
        """
        logger.info("Starting fine-tuning pipeline...")
        
        # Step 1: Parse data
        logger.info("Step 1: Parsing training data...")
        from parse_data import DataParser
        
        parser = DataParser(data_dir)
        parsed_data = parser.parse_all_files()
        
        if not parsed_data:
            raise ValueError("No data found to parse")
        
        # Save parsed data
        parsed_path = self.output_dir / "parsed_data.json"
        parser.save_parsed_data(parsed_data, str(parsed_path))
        
        # Step 2: Prepare dataset
        logger.info("Step 2: Preparing training dataset...")
        from prepare_dataset import DatasetPreparer
        
        preparer = DatasetPreparer()
        train_data_path = self.output_dir / "train.jsonl"
        preparer.prepare_training_data(parsed_data, str(train_data_path), max_entries=max_entries)
        
        # Create validation split
        val_data_path = self.output_dir / "train_val.jsonl"
        preparer.create_validation_split(str(train_data_path))
        
        # Step 3: Load model and tokenizer
        logger.info("Step 3: Loading model and tokenizer...")
        model, tokenizer, config = self.load_model_and_tokenizer()
        
        # Step 4: Train LoRA adapters
        logger.info("Step 4: Training LoRA adapters...")
        adapter_path = self.train_lora(
            model, tokenizer, str(train_data_path), str(val_data_path),
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            lora_rank=lora_rank
        )
        
        # Step 5: Merge adapters
        logger.info("Step 5: Merging LoRA adapters...")
        merged_path = self.merge_lora_adapters(adapter_path)
        
        # Step 6: Convert to GGUF (optional)
        gguf_path = None
        if convert_to_gguf:
            logger.info("Step 6: Converting to GGUF...")
            gguf_path = self.convert_to_gguf(merged_path)
        
        # Return results
        results = {
            "parsed_data": str(parsed_path),
            "train_data": str(train_data_path),
            "val_data": str(val_data_path),
            "adapter_path": adapter_path,
            "merged_model": merged_path
        }
        
        if gguf_path:
            results["gguf_model"] = gguf_path
        
        logger.info("Fine-tuning pipeline completed successfully!")
        return results

def main():
    """Main function to run fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-27B-it-Abliterated model")
    
    parser.add_argument("--model_path", required=True,
                       help="Path to Gemma-3-27B-it-Abliterated model (GGUF or MLX)")
    parser.add_argument("--vision_model_path", 
                       help="Path to vision model for PDF processing")
    parser.add_argument("--data_dir", default="~/Desktop/trainingdata/",
                       help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--max_entries", type=int,
                       help="Maximum number of entries to process")
    parser.add_argument("--convert_to_gguf", action="store_true",
                       help="Convert final model to GGUF format")
    
    args = parser.parse_args()
    
    try:
        # Initialize fine-tuner
        tuner = GemmaFineTuner(args.model_path, args.vision_model_path)
        
        # Run pipeline
        results = tuner.run_finetuning_pipeline(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            max_entries=args.max_entries,
            convert_to_gguf=args.convert_to_gguf
        )
        
        # Print results
        logger.info("Fine-tuning completed successfully!")
        logger.info("Output files:")
        for key, path in results.items():
            logger.info(f"  {key}: {path}")
        
        # Save results summary
        with open("output/finetune_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 