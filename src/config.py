"""
Configuration module for Hebrew Nikud training.

Provides default configuration values and argparse setup for training parameters.
"""

import argparse
from pathlib import Path


class Config:
    """Training configuration with defaults."""
    
    # Dataset
    train_file: str = "data/train.txt"
    eval_ratio: float = 0.2
    seed: int = 42
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_epochs: int = 10
    
    # Model
    model_name: str = "dicta-il/dictabert-large-char"
    dropout: float = 0.1
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    
    # Device
    device: str = None  # None for auto-detect
    
    # Wandb
    wandb_mode: str = "offline"
    wandb_project: str = "hebrew-nikud"
    wandb_run_name: str = None
    
    def __init__(self, **kwargs):
        """Initialize config with optional overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_args(cls):
        """Parse arguments and create config."""
        parser = argparse.ArgumentParser(
            description="Train Hebrew Nikud BERT model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Dataset arguments
        parser.add_argument("--train-file", type=str, default=cls.train_file,
                          help="Path to training data file")
        parser.add_argument("--eval-ratio", type=float, default=cls.eval_ratio,
                          help="Ratio of data to use for evaluation")
        parser.add_argument("--seed", type=int, default=cls.seed,
                          help="Random seed for reproducibility")
        
        # Training arguments
        parser.add_argument("--batch-size", type=int, default=cls.batch_size,
                          help="Batch size for training")
        parser.add_argument("--lr", type=float, default=cls.learning_rate,
                          help="Learning rate")
        parser.add_argument("--max-epochs", type=int, default=cls.max_epochs,
                          help="Maximum number of training epochs")
        
        # Model arguments
        parser.add_argument("--model-name", type=str, default=cls.model_name,
                          help="Pretrained model name")
        parser.add_argument("--dropout", type=float, default=cls.dropout,
                          help="Dropout rate")
        
        # Checkpoint arguments
        parser.add_argument("--checkpoint-dir", type=str, default=cls.checkpoint_dir,
                          help="Directory to save checkpoints")
        
        # Device
        parser.add_argument("--device", type=str, default=cls.device,
                          help="Device to train on (cpu/cuda/mps, None for auto)")
        
        # Wandb arguments
        parser.add_argument("--wandb-mode", type=str, default=cls.wandb_mode,
                          choices=["online", "offline", "disabled"],
                          help="Wandb logging mode")
        parser.add_argument("--wandb-project", type=str, default=cls.wandb_project,
                          help="Wandb project name")
        parser.add_argument("--wandb-run-name", type=str, default=cls.wandb_run_name,
                          help="Wandb run name")
        
        args = parser.parse_args()
        
        # Create config from args
        config = cls(
            train_file=args.train_file,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_epochs=args.max_epochs,
            model_name=args.model_name,
            dropout=args.dropout,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        
        return config
    
    def __repr__(self):
        """String representation of config."""
        lines = ["Configuration:"]
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

