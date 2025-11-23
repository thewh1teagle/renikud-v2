"""
Evaluation module for Hebrew Nikud model.

Calculates metrics including WER (Word Error Rate), CER (Character Error Rate),
and per-task accuracies.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict
from decode import reconstruct_text_from_predictions
from tqdm import tqdm
import unicodedata
import jiwer


def calculate_cer(predicted_text: str, target_text: str) -> float:
    # Apply same normalization as training data
    from normalize import normalize
    predicted_text = normalize(predicted_text)
    target_text = normalize(target_text)
    
    # Handle empty strings
    if not target_text:
        return 0.0 if not predicted_text else 1.0
    
    # jiwer expects (reference, hypothesis) order
    return jiwer.cer(target_text, predicted_text)


def calculate_wer(predicted_text: str, target_text: str) -> float:
    # Apply same normalization as training data
    from normalize import normalize
    predicted_text = normalize(predicted_text)
    target_text = normalize(target_text)
    
    # Handle empty strings
    if not target_text.strip():
        return 0.0 if not predicted_text.strip() else 1.0
    
    # jiwer expects (reference, hypothesis) order
    return jiwer.wer(target_text, predicted_text)


def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    tokenizer,
    desc: str = "Evaluating"
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
        
    Returns:
        Dictionary with metrics: loss, wer, cer, and per-task accuracies
    """
    model.eval()
    
    total_loss = 0.0
    total_vowel_loss = 0.0
    total_dagesh_loss = 0.0
    total_sin_loss = 0.0
    total_stress_loss = 0.0
    
    # Accuracy tracking
    vowel_correct = 0
    vowel_total = 0
    dagesh_correct = 0
    dagesh_total = 0
    sin_correct = 0
    sin_total = 0
    stress_correct = 0
    stress_total = 0
    
    # WER/CER tracking
    total_wer = 0.0
    total_cer = 0.0
    num_samples = 0
    
    # Flag to print first sample comparison
    printed_first_sample = False
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
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
            )
            
            # Accumulate losses
            total_loss += outputs['loss'].item()
            total_vowel_loss += outputs['vowel_loss'].item()
            total_dagesh_loss += outputs['dagesh_loss'].item()
            total_sin_loss += outputs['sin_loss'].item()
            total_stress_loss += outputs['stress_loss'].item()
            
            # Get predictions
            vowel_preds = torch.argmax(outputs['vowel_logits'], dim=-1)
            dagesh_preds = (torch.sigmoid(outputs['dagesh_logits']) > 0.5).long()
            sin_preds = (torch.sigmoid(outputs['sin_logits']) > 0.5).long()
            stress_preds = (torch.sigmoid(outputs['stress_logits']) > 0.5).long()
            
            # Calculate accuracies (only on non-ignored positions)
            vowel_mask = vowel_labels != -100
            vowel_correct += (vowel_preds[vowel_mask] == vowel_labels[vowel_mask]).sum().item()
            vowel_total += vowel_mask.sum().item()
            
            dagesh_mask = dagesh_labels != -100
            dagesh_correct += (dagesh_preds[dagesh_mask] == dagesh_labels[dagesh_mask]).sum().item()
            dagesh_total += dagesh_mask.sum().item()
            
            sin_mask = sin_labels != -100
            sin_correct += (sin_preds[sin_mask] == sin_labels[sin_mask]).sum().item()
            sin_total += sin_mask.sum().item()
            
            stress_mask = stress_labels != -100
            stress_correct += (stress_preds[stress_mask] == stress_labels[stress_mask]).sum().item()
            stress_total += stress_mask.sum().item()
            
            # Calculate WER/CER for each sample in batch
            for i in range(input_ids.shape[0]):
                predicted_text = reconstruct_text_from_predictions(
                    input_ids[i],
                    vowel_preds[i],
                    dagesh_preds[i],
                    sin_preds[i],
                    stress_preds[i],
                    tokenizer
                )
                
                target_text = batch['original_text'][i]
                
                total_wer += calculate_wer(predicted_text, target_text)
                total_cer += calculate_cer(predicted_text, target_text)
                num_samples += 1
                
                # Print first sample comparison for debugging
                if not printed_first_sample:
                    print("\n" + "="*80)
                    print("SAMPLE COMPARISON (First evaluation sample):")
                    print("="*80)
                    print(f"Original:  {target_text}")
                    print(f"Predicted: {predicted_text}")
                    print(f"Match: {target_text == predicted_text}")
                    print("="*80 + "\n")
                    printed_first_sample = True
    
    # Calculate averages
    num_batches = len(dataloader)
    
    # Handle empty dataloader
    if num_batches == 0:
        return {
            'loss': 0.0,
            'vowel_loss': 0.0,
            'dagesh_loss': 0.0,
            'sin_loss': 0.0,
            'stress_loss': 0.0,
            'vowel_acc': 0.0,
            'dagesh_acc': 0.0,
            'sin_acc': 0.0,
            'stress_acc': 0.0,
            'wer': 0.0,
            'cer': 0.0,
        }
    
    avg_loss = total_loss / num_batches
    avg_vowel_loss = total_vowel_loss / num_batches
    avg_dagesh_loss = total_dagesh_loss / num_batches
    avg_sin_loss = total_sin_loss / num_batches
    avg_stress_loss = total_stress_loss / num_batches
    
    vowel_acc = vowel_correct / vowel_total if vowel_total > 0 else 0.0
    dagesh_acc = dagesh_correct / dagesh_total if dagesh_total > 0 else 0.0
    sin_acc = sin_correct / sin_total if sin_total > 0 else 0.0
    stress_acc = stress_correct / stress_total if stress_total > 0 else 0.0
    
    avg_wer = total_wer / num_samples if num_samples > 0 else 0.0
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'vowel_loss': avg_vowel_loss,
        'dagesh_loss': avg_dagesh_loss,
        'sin_loss': avg_sin_loss,
        'stress_loss': avg_stress_loss,
        'vowel_acc': vowel_acc,
        'dagesh_acc': dagesh_acc,
        'sin_acc': sin_acc,
        'stress_acc': stress_acc,
        'wer': avg_wer,
        'cer': avg_cer,
    }
