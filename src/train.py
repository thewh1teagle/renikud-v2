"""
Training script for Hebrew Nikud BERT model with HuggingFace Trainer.
"""

import torch
from transformers import AutoTokenizer, TrainingArguments
from pathlib import Path

from model import HebrewNikudModel, count_parameters
from dataset import (
    NikudDataset,
    load_dataset_from_file,
    split_dataset,
    collate_fn
)
from train_loop import NikudTrainer, set_seed
from config import Config
import wandb


def main():
    """Main training function with HuggingFace Trainer."""
    # Parse configuration
    config = Config.from_args()
    
    print("=" * 80)
    print("Hebrew Nikud Training (HuggingFace Trainer)")
    print("=" * 80)
    print(config)
    print("=" * 80)
    
    # Set random seed
    set_seed(config.seed)
    print(f"\n✓ Random seed set to {config.seed}")
    
    # Set device
    if config.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = config.device
    print(f"✓ Using device: {device}")
    
    # Initialize wandb if needed
    if config.wandb_mode != "disabled":
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            mode=config.wandb_mode,
            config=vars(config)
        )
        print(f"✓ Wandb initialized (mode: {config.wandb_mode})")
    else:
        print("✓ Wandb disabled")
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    print("✓ Tokenizer loaded")
    
    # Load and split dataset
    print("\n" + "=" * 80)
    print(f"Loading dataset from {config.train_file}...")
    texts = load_dataset_from_file(config.train_file)
    print(f"✓ Loaded {len(texts)} texts")
    
    print(f"\nSplitting dataset (eval_max_lines={config.eval_max_lines})...")
    train_texts, eval_texts = split_dataset(texts, config.eval_max_lines, config.seed)
    print(f"✓ Train: {len(train_texts)} texts")
    print(f"✓ Eval: {len(eval_texts)} texts")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NikudDataset(train_texts, tokenizer)
    eval_dataset = NikudDataset(eval_texts, tokenizer)
    print("✓ Datasets created")
    
    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing model...")
    model = HebrewNikudModel(model_name=config.model_name, dropout=config.dropout)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print("✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.2f} MB")
    
    # Log to wandb
    if config.wandb_mode != "disabled":
        wandb.config.update({
            'total_params': total_params,
            'trainable_params': trainable_params
        })
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir,
        num_train_epochs=config.max_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=config.save_best,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_dir=f"{config.checkpoint_dir}/logs",
        logging_strategy="steps",  # Log during training, not just at epoch end
        logging_steps=50,  # Log every 50 steps
        report_to="wandb" if config.wandb_mode != "disabled" else "none",
        seed=config.seed,
        dataloader_pin_memory=False if device == 'mps' else True,
        remove_unused_columns=False,  # Keep original_text for WER/CER calculation
    )
    
    # Create trainer
    print("\n" + "=" * 80)
    print("Setting up trainer...")
    trainer = NikudTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    print("✓ Trainer ready")
    
    # Train
    print("\n" + "=" * 80)
    print(f"Starting training for {config.max_epochs} epochs")
    print("=" * 80)
    
    trainer.train()
    
    # Save final model
    final_path = Path(config.checkpoint_dir) / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Final model saved to {final_path}")
    
    # Get best metrics
    best_metrics = trainer.state.best_metric
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("Best Evaluation Metrics:")
    if best_metrics is not None:
        print(f"  CER: {best_metrics:.4f}")
    print("=" * 80)
    
    # Finish wandb
    if config.wandb_mode != "disabled":
        wandb.finish()


if __name__ == '__main__':
    main()
