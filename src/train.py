"""
Training script for Hebrew Nikud BERT model.

This script trains the model on a single example to verify it can learn.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from pathlib import Path

from model import HebrewNikudModel
from dataset import NikudDataset, prepare_training_data


def train_single_example(
    nikud_text: str = "הַאִיש שֵלֹא רַצַה לִהיוֹת אַגַדַה",
    num_epochs: int = 500,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    device: str = None
):
    """
    Train the model on a single example to verify it can overfit.
    
    Args:
        nikud_text: Training text with nikud marks
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on (None for auto-detect)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char')
    
    # Prepare training data
    print(f"Preparing training data: {nikud_text}")
    train_data = prepare_training_data(nikud_text, tokenizer)
    
    print(f"Plain text: {train_data['plain_text']}")
    print(f"Input IDs shape: {train_data['input_ids'].shape}")
    print(f"Vowel labels: {train_data['vowel_labels']}")
    print(f"Dagesh labels: {train_data['dagesh_labels']}")
    print(f"Sin labels: {train_data['sin_labels']}")
    print(f"Stress labels: {train_data['stress_labels']}")
    
    # Initialize model
    print("\nInitializing model...")
    model = HebrewNikudModel()
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 80)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Move data to device
        input_ids = train_data['input_ids'].unsqueeze(0).to(device)
        attention_mask = train_data['attention_mask'].unsqueeze(0).to(device)
        vowel_labels = train_data['vowel_labels'].unsqueeze(0).to(device)
        dagesh_labels = train_data['dagesh_labels'].unsqueeze(0).to(device)
        sin_labels = train_data['sin_labels'].unsqueeze(0).to(device)
        stress_labels = train_data['stress_labels'].unsqueeze(0).to(device)
        
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
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Total Loss: {loss.item():.4f}")
            print(f"  Vowel Loss: {outputs['vowel_loss'].item():.4f}")
            print(f"  Dagesh Loss: {outputs['dagesh_loss'].item():.4f}")
            print(f"  Sin Loss: {outputs['sin_loss'].item():.4f}")
            print(f"  Stress Loss: {outputs['stress_loss'].item():.4f}")
            print()
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {checkpoint_dir}")
    
    # Test prediction on training example
    print("\n" + "=" * 80)
    print("Testing prediction on training example...")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        input_ids = train_data['input_ids'].unsqueeze(0).to(device)
        attention_mask = train_data['attention_mask'].unsqueeze(0).to(device)
        
        predictions = model.predict(input_ids, attention_mask, tokenizer)
        
        # Compare with ground truth
        print(f"\nOriginal text: {nikud_text}")
        print(f"Plain text: {train_data['plain_text']}")
        print()
        
        vowel_labels = train_data['vowel_labels'].unsqueeze(0)
        dagesh_labels = train_data['dagesh_labels'].unsqueeze(0)
        sin_labels = train_data['sin_labels'].unsqueeze(0)
        stress_labels = train_data['stress_labels'].unsqueeze(0)
        
        # Calculate accuracy for non-ignored positions
        vowel_mask = vowel_labels != -100
        dagesh_mask = dagesh_labels != -100
        sin_mask = sin_labels != -100
        stress_mask = stress_labels != -100
        
        vowel_correct = (predictions['vowel'][vowel_mask].cpu() == vowel_labels[vowel_mask]).sum().item()
        vowel_total = vowel_mask.sum().item()
        vowel_acc = vowel_correct / vowel_total if vowel_total > 0 else 0
        
        dagesh_correct = (predictions['dagesh'][dagesh_mask].cpu() == dagesh_labels[dagesh_mask]).sum().item()
        dagesh_total = dagesh_mask.sum().item()
        dagesh_acc = dagesh_correct / dagesh_total if dagesh_total > 0 else 0
        
        sin_correct = (predictions['sin'][sin_mask].cpu() == sin_labels[sin_mask]).sum().item()
        sin_total = sin_mask.sum().item()
        sin_acc = sin_correct / sin_total if sin_total > 0 else 0
        
        stress_correct = (predictions['stress'][stress_mask].cpu() == stress_labels[stress_mask]).sum().item()
        stress_total = stress_mask.sum().item()
        stress_acc = stress_correct / stress_total if stress_total > 0 else 0
        
        print(f"Vowel Accuracy: {vowel_correct}/{vowel_total} = {vowel_acc:.2%}")
        print(f"Dagesh Accuracy: {dagesh_correct}/{dagesh_total} = {dagesh_acc:.2%}")
        print(f"Sin Accuracy: {sin_correct}/{sin_total} = {sin_acc:.2%}")
        print(f"Stress Accuracy: {stress_correct}/{stress_total} = {stress_acc:.2%}")
        
        # Detailed comparison
        print("\nDetailed token-by-token comparison:")
        print("-" * 80)
        
        for i in range(1, input_ids.shape[1] - 1):  # Skip [CLS] and [SEP]
            token_id = input_ids[0, i].item()
            token_char = tokenizer.decode([token_id])
            
            if vowel_labels[0, i].item() != -100:
                true_vowel = vowel_labels[0, i].item()
                pred_vowel = predictions['vowel'][0, i].item()
                vowel_match = "✓" if true_vowel == pred_vowel else "✗"
                
                true_dagesh = dagesh_labels[0, i].item()
                pred_dagesh = predictions['dagesh'][0, i].item()
                dagesh_match = "✓" if true_dagesh == pred_dagesh else "✗"
                
                true_sin = sin_labels[0, i].item()
                pred_sin = predictions['sin'][0, i].item()
                sin_match = "✓" if true_sin == pred_sin else "✗"
                
                true_stress = stress_labels[0, i].item()
                pred_stress = predictions['stress'][0, i].item()
                stress_match = "✓" if true_stress == pred_stress else "✗"
                
                print(f"Char '{token_char}': V:{pred_vowel}{vowel_match} D:{pred_dagesh}{dagesh_match} "
                      f"S:{pred_sin}{sin_match} St:{pred_stress}{stress_match}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Hebrew Nikud BERT model')
    parser.add_argument('--text', type=str, 
                       default="הַאִיש שֵלֹא רַצַה לִהיוֹת אַגַדַה",
                       help='Training text with nikud')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    train_single_example(
        nikud_text=args.text,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()

