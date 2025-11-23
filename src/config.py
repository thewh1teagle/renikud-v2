"""
Configuration module for Hebrew Nikud training.

Provides argparse setup for training parameters.
"""

import argparse


def get_args():
    """Parse command-line arguments and return configuration namespace."""
    parser = argparse.ArgumentParser(
        description='Train Hebrew Nikud BERT model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--train-file', type=str, default='data/train.txt',
                        help='Path to training data file')
    parser.add_argument('--eval-max-lines', type=int, default=100,
                        help='Maximum number of lines to use for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='Maximum number of training epochs')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='dicta-il/dictabert-large-char',
                        help='Pretrained model name')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='Save best model based on validation metrics')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cpu/cuda/mps, None for auto)')
    
    # Wandb arguments
    parser.add_argument('--wandb-mode', type=str, default='disabled',
                        choices=['online', 'offline', 'disabled'],
                        help='Wandb logging mode')
    parser.add_argument('--wandb-project', type=str, default='hebrew-nikud',
                        help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name')
    
    args = parser.parse_args()
    
    # Normalize arg names (replace - with _)
    args.train_file = args.train_file
    args.eval_max_lines = args.eval_max_lines
    args.learning_rate = args.lr
    args.max_epochs = args.max_epochs
    args.max_grad_norm = args.max_grad_norm
    args.model_name = args.model_name
    args.batch_size = args.batch_size
    args.checkpoint_dir = args.checkpoint_dir
    args.save_best = args.save_best
    args.wandb_mode = args.wandb_mode
    args.wandb_project = args.wandb_project
    args.wandb_run_name = args.wandb_run_name
    
    return args
