"""
Training script for Hebrew Nikud BERT model with HuggingFace Trainer.
"""

import torch
from transformers import AutoTokenizer, TrainingArguments
from pathlib import Path

from model import HebrewNikudModel, count_parameters
from dataset import NikudDataset, load_dataset_from_file, split_dataset, collate_fn
from trainer import NikudTrainer
from config import get_args


def main():
    """Main training function with HuggingFace Trainer."""
    # Parse configuration
    config = get_args()
    
    print("=" * 80)
    print("Hebrew Nikud Training")
    print("=" * 80)
    print("Configuration:")
    for key, value in vars(config).items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load and split dataset
    texts = load_dataset_from_file(config.train_file)
    train_texts, eval_texts = split_dataset(texts, config.eval_max_lines, config.seed)
    print(f"Loaded {len(texts)} texts ({len(train_texts)} train, {len(eval_texts)} eval)")
    
    # Create datasets
    train_dataset = NikudDataset(train_texts, tokenizer)
    eval_dataset = NikudDataset(eval_texts, tokenizer)
    
    # Initialize model
    print("\nInitializing model...")
    model = HebrewNikudModel(model_name=config.model_name, dropout=config.dropout)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {total_params:,} parameters (~{total_params * 4 / (1024**2):.1f} MB)")
    
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
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=config.save_best,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_dir=f"{config.checkpoint_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb" if config.wandb_mode != "disabled" else "none",
        seed=config.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    
    # Create trainer
    print(f"\nStarting training for {config.max_epochs} epochs...")
    print("=" * 80)
    trainer = NikudTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    final_path = Path(config.checkpoint_dir) / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    
    # Print final summary
    best_metrics = trainer.state.best_metric
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Final model: {final_path}")
    if best_metrics is not None:
        print(f"Best CER: {best_metrics:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
