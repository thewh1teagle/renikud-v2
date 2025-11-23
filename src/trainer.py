"""
Custom Trainer for Hebrew Nikud model with WER/CER metrics.
"""

import torch
from transformers import Trainer
from evaluate import calculate_wer, calculate_cer
from decode import reconstruct_text_from_predictions


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

