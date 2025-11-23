"""
Training script for Hebrew Nikud BERT model with HuggingFace Trainer.
"""

import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from pathlib import Path

from model import HebrewNikudModel, count_parameters
from dataset import NikudDataset, load_dataset_from_file, split_dataset, collate_fn
from evaluate import calculate_wer, calculate_cer, reconstruct_text_from_predictions
from config import get_args


class NikudTrainer(Trainer):
    """Custom Trainer for Hebrew Nikud model with WER/CER metrics."""
    
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_class = tokenizer
        self.tokenizer = tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for training."""
        # Extract all labels
        vowel_labels = inputs.pop("vowel_labels")
        dagesh_labels = inputs.pop("dagesh_labels")
        sin_labels = inputs.pop("sin_labels")
        stress_labels = inputs.pop("stress_labels")
        
        # Remove non-tensor fields
        inputs.pop("plain_text", None)
        inputs.pop("original_text", None)
        
        outputs = model(
            **inputs,
            vowel_labels=vowel_labels,
            dagesh_labels=dagesh_labels,
            sin_labels=sin_labels,
            stress_labels=stress_labels
        )
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation loop that includes WER/CER calculations."""
        # Call parent evaluation
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Calculate WER/CER
        model = self.model
        model.eval()
        
        total_wer = 0.0
        total_cer = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                
                # Get predictions
                predictions = model.predict(input_ids, attention_mask)
                
                # Calculate WER/CER for each sample
                for i in range(input_ids.shape[0]):
                    predicted_text = reconstruct_text_from_predictions(
                        input_ids[i],
                        predictions['vowel'][i],
                        predictions['dagesh'][i],
                        predictions['sin'][i],
                        predictions['stress'][i],
                        self.processing_class
                    )
                    
                    target_text = batch['original_text'][i]
                    
                    total_wer += calculate_wer(predicted_text, target_text)
                    total_cer += calculate_cer(predicted_text, target_text)
                    num_samples += 1
        
        # Add WER/CER to metrics
        if num_samples > 0:
            output.metrics[f'{metric_key_prefix}_wer'] = total_wer / num_samples
            output.metrics[f'{metric_key_prefix}_cer'] = total_cer / num_samples
        
        return output


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
        save_strategy="epoch",
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
        tokenizer=tokenizer,
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
