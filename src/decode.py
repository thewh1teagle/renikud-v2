"""
Utility functions for reconstructing Hebrew text with nikud from predictions.
"""

import torch
import unicodedata


def reconstruct_text_from_predictions(
    input_ids: torch.Tensor,
    vowel_preds: torch.Tensor,
    dagesh_preds: torch.Tensor,
    sin_preds: torch.Tensor,
    stress_preds: torch.Tensor,
    tokenizer
) -> str:
    """
    Reconstruct Hebrew text with nikud marks from model predictions.
    
    Args:
        input_ids: Token IDs [seq_len]
        vowel_preds: Vowel class predictions [seq_len] (0-5)
        dagesh_preds: Dagesh binary predictions [seq_len] (0/1)
        sin_preds: Sin binary predictions [seq_len] (0/1)
        stress_preds: Stress binary predictions [seq_len] (0/1)
        tokenizer: Tokenizer for decoding token IDs
        
    Returns:
        Text with predicted nikud marks
    """
    # Import here to avoid circular dependencies
    from dataset import ID_TO_VOWEL
    from constants import DAGESH, S_SIN, STRESS_HATAMA, CAN_HAVE_DAGESH, CAN_HAVE_SIN, LETTERS
    from normalize import normalize
    result = []
    
    # Get special token IDs
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Skip [CLS] (position 0) and stop at [SEP] or [PAD]
    for i in range(1, len(input_ids)):
        token_id = input_ids[i].item()
        
        # Stop at special tokens
        if token_id == sep_token_id or token_id == pad_token_id:
            break
        
        char = tokenizer.decode([token_id])
        result.append(char)
        
        # Only add nikud marks for Hebrew letters
        if char not in LETTERS:
            continue
        
        diacritics = []
        
        # Add vowel (if not VOWEL_NONE)
        vowel_id = vowel_preds[i].item()
        if vowel_id > 0:  # 0 is VOWEL_NONE
            vowel_char = ID_TO_VOWEL.get(vowel_id)
            if vowel_char:
                diacritics.append(vowel_char)
        
        # Add dagesh if predicted and character supports it
        if dagesh_preds[i].item() == 1 and char in CAN_HAVE_DAGESH:
            diacritics.append(DAGESH)
        
        # Add sin if predicted and character supports it
        if sin_preds[i].item() == 1 and char in CAN_HAVE_SIN:
            diacritics.append(S_SIN)
        
        # Add stress if predicted
        if stress_preds[i].item() == 1:
            diacritics.append(STRESS_HATAMA)
        
        # Sort diacritics for canonical order
        diacritics.sort()
        result.extend(diacritics)
    
    # Combine and normalize
    text = ''.join(result)
    text = normalize(text)  # Apply same normalization as training data
    text = unicodedata.normalize('NFC', text)
    
    return text

