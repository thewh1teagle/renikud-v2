"""
Training loop for Hebrew Nikud BERT model.

Orchestrates the full training pipeline including:
- Data loading and splitting
- Model initialization and training
- Evaluation with WER/CER metrics
- Checkpoint saving
- Wandb logging
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np

from config import Config
from model import HebrewNikudModel, count_parameters
from dataset import (
    NikudDataset,
    load_dataset_from_file,
    split_dataset,
    collate_fn
)
from evaluate import evaluate


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device: str,
    tokenizer,
    epoch: int,
    total_epochs: int
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: HebrewNikudModel instance
        dataloader: Training DataLoader
        optimizer: Optimizer
        device: Device to train on
        tokenizer: Tokenizer for the model
        epoch: Current epoch number
        total_epochs: Total number of epochs
        
    Returns:
        Dictionary with average losses
    """
    model.train()
    
    total_loss = 0.0
    total_vowel_loss = 0.0
    total_dagesh_loss = 0.0
    total_sin_loss = 0.0
    total_stress_loss = 0.0
    
    # Progress bar
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=True
    )
    
    for batch in pbar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        vowel_labels = batch['vowel_labels'].to(device)
        dagesh_labels = batch['dagesh_labels'].to(device)
        sin_labels = batch['sin_labels'].to(device)
        stress_labels = batch['stress_labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vowel_labels=vowel_labels,
            dagesh_labels=dagesh_labels,
            sin_labels=sin_labels,
            stress_labels=stress_labels,
            tokenizer=tokenizer
        )
        
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_vowel_loss += outputs['vowel_loss'].item()
        total_dagesh_loss += outputs['dagesh_loss'].item()
        total_sin_loss += outputs['sin_loss'].item()
        total_stress_loss += outputs['stress_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'vowel': f"{outputs['vowel_loss'].item():.4f}",
            'dagesh': f"{outputs['dagesh_loss'].item():.4f}",
        })
    
    # Calculate averages
    num_batches = len(dataloader)
    return {
        'train_loss': total_loss / num_batches,
        'train_vowel_loss': total_vowel_loss / num_batches,
        'train_dagesh_loss': total_dagesh_loss / num_batches,
        'train_sin_loss': total_sin_loss / num_batches,
        'train_stress_loss': total_stress_loss / num_batches,
    }


def main():
    """Main training function."""
    # Parse configuration
    config = Config.from_args()
    
    print("=" * 80)
    print("Hebrew Nikud Training")
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
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=vars(config)
    )
    print(f"✓ Wandb initialized (mode: {config.wandb_mode})")
    
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
    
    print(f"\nSplitting dataset (eval_ratio={config.eval_ratio})...")
    train_texts, eval_texts = split_dataset(texts, config.eval_ratio, config.seed)
    print(f"✓ Train: {len(train_texts)} texts")
    print(f"✓ Eval: {len(eval_texts)} texts")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NikudDataset(train_texts, tokenizer)
    eval_dataset = NikudDataset(eval_texts, tokenizer)
    print("✓ Datasets created")
    
    # Create dataloaders
    print(f"\nCreating dataloaders (batch_size={config.batch_size})...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    print("✓ Dataloaders created")
    
    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing model...")
    model = HebrewNikudModel(model_name=config.model_name, dropout=config.dropout)
    model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.2f} MB")
    
    # Log to wandb
    wandb.config.update({
        'total_params': total_params,
        'trainable_params': trainable_params
    })
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 80)
    print(f"Starting training for {config.max_epochs} epochs")
    print("=" * 80)
    
    best_eval_loss = float('inf')
    best_wer = float('inf')
    best_cer = float('inf')
    
    for epoch in range(1, config.max_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, tokenizer,
            epoch, config.max_epochs
        )
        
        # Evaluate
        eval_metrics = evaluate(
            model, eval_loader, device, tokenizer,
            desc=f"Evaluating Epoch {epoch}"
        )
        
        # Add 'eval_' prefix to eval metrics
        eval_metrics_prefixed = {f'eval_{k}': v for k, v in eval_metrics.items()}
        
        # Log to wandb
        wandb.log({
            **train_metrics,
            **eval_metrics_prefixed,
            'epoch': epoch,
            'learning_rate': config.learning_rate
        })
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Eval Loss:  {eval_metrics['loss']:.4f}")
        print(f"  Eval WER:   {eval_metrics['wer']:.4f}")
        print(f"  Eval CER:   {eval_metrics['cer']:.4f}")
        print(f"  Vowel Acc:  {eval_metrics['vowel_acc']:.4f}")
        print(f"  Dagesh Acc: {eval_metrics['dagesh_acc']:.4f}")
        print(f"  Sin Acc:    {eval_metrics['sin_acc']:.4f}")
        print(f"  Stress Acc: {eval_metrics['stress_acc']:.4f}")
        
        # Save best model
        if config.save_best and eval_metrics['loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['loss']
            best_wer = eval_metrics['wer']
            best_cer = eval_metrics['cer']
            
            checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (loss: {best_eval_loss:.4f})")
        
        print("-" * 80)
    
    # Save final model
    final_path = Path(config.checkpoint_dir) / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Final model saved to {final_path}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Evaluation Metrics:")
    print(f"  Loss: {best_eval_loss:.4f}")
    print(f"  WER:  {best_wer:.4f}")
    print(f"  CER:  {best_cer:.4f}")
    print("=" * 80)
    
    # Finish wandb
    wandb.finish()


if __name__ == '__main__':
    main()

