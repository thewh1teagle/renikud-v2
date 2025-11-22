"""
Inference script for Hebrew Nikud BERT model.

This module handles loading the trained model and generating nikud predictions.
"""

import torch
from transformers import AutoTokenizer
from typing import List, Tuple
import unicodedata

from model import HebrewNikudModel, load_model
from dataset import ID_TO_VOWEL
from constants import (
    A_PATAH, E_TSERE, I_HIRIK, O_HOLAM, U_QUBUT,
    DAGESH, S_SIN, STRESS_HATAMA,
    CAN_HAVE_DAGESH, CAN_HAVE_SIN, LETTERS
)


class NikudPredictor:
    """Predictor class for adding nikud to Hebrew text."""
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on (None for auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.device = device
        print(f"Loading model on device: {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char')
        
        # Load model
        self.model = HebrewNikudModel()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict(self, text: str) -> str:
        """
        Add nikud marks to Hebrew text.
        
        Args:
            text: Plain Hebrew text without nikud (can contain mixed content)
            
        Returns:
            Text with predicted nikud marks on Hebrew letters, other characters preserved
        """
        # Don't filter! Keep the original text as-is
        if not text.strip():
            return text
        
        # Tokenize the full text (tokenizer will handle all characters)
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=False,
            truncation=False,
            add_special_tokens=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict(input_ids, attention_mask, self.tokenizer)
        
        # Reconstruct text with nikud
        nikud_text = self._reconstruct_text(
            input_ids[0],
            predictions['vowel'][0],
            predictions['dagesh'][0],
            predictions['sin'][0],
            predictions['stress'][0]
        )
        
        return nikud_text
    
    def _reconstruct_text(
        self,
        input_ids: torch.Tensor,
        vowel_preds: torch.Tensor,
        dagesh_preds: torch.Tensor,
        sin_preds: torch.Tensor,
        stress_preds: torch.Tensor
    ) -> str:
        """
        Reconstruct Hebrew text with nikud marks from predictions.
        
        Args:
            input_ids: Token IDs
            vowel_preds: Predicted vowel labels
            dagesh_preds: Predicted dagesh labels
            sin_preds: Predicted sin labels
            stress_preds: Predicted stress labels
            
        Returns:
            Text with nikud marks
        """
        result = []
        
        # Skip [CLS] token at position 0 and [SEP] at the end
        for i in range(1, len(input_ids) - 1):
            token_id = input_ids[i].item()
            char = self.tokenizer.decode([token_id])
            
            # Add base character
            result.append(char)
            
            # Only add nikud marks for Hebrew letters (skip spaces, punctuation, etc.)
            if char not in LETTERS:
                continue
            
            # Add predicted nikud marks
            diacritics = []
            
            # Add vowel if predicted
            vowel_id = vowel_preds[i].item()
            if vowel_id > 0:  # 0 means no vowel
                vowel_char = ID_TO_VOWEL.get(vowel_id)
                if vowel_char:
                    diacritics.append(vowel_char)
            
            # Add dagesh if predicted and valid
            if dagesh_preds[i].item() == 1 and char in CAN_HAVE_DAGESH:
                diacritics.append(DAGESH)
            
            # Add sin if predicted and valid
            if sin_preds[i].item() == 1 and char in CAN_HAVE_SIN:
                diacritics.append(S_SIN)
            
            # Add stress if predicted
            if stress_preds[i].item() == 1:
                diacritics.append(STRESS_HATAMA)
            
            # Sort diacritics (canonical order)
            diacritics.sort()
            result.extend(diacritics)
        
        # Normalize to combine characters properly
        text = ''.join(result)
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Add nikud marks to multiple Hebrew texts.
        
        Args:
            texts: List of plain Hebrew texts
            
        Returns:
            List of texts with predicted nikud marks
        """
        return [self.predict(text) for text in texts]


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict nikud for Hebrew text')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to add nikud to')
    parser.add_argument('--file', type=str, default=None,
                       help='File containing text to add nikud to')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NikudPredictor(args.checkpoint, device=args.device)
    
    # Get input text
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("Interactive mode. Enter Hebrew text (Ctrl+C to exit):")
        texts = []
        try:
            while True:
                text = input("> ")
                if text.strip():
                    texts.append(text.strip())
        except KeyboardInterrupt:
            print("\nExiting...")
    
    # Generate predictions
    print("\n" + "=" * 80)
    print("Predictions:")
    print("=" * 80)
    
    for i, text in enumerate(texts):
        nikud_text = predictor.predict(text)
        print(f"\nInput:  {text}")
        print(f"Output: {nikud_text}")
        
        if i < len(texts) - 1:
            print("-" * 80)


if __name__ == '__main__':
    main()

