"""
Prediction logic for Hebrew Nikud model.
"""

import numpy as np
import unicodedata
from pathlib import Path

from .constants import (
    ID_TO_VOWEL, DAGESH, S_SIN, STRESS_HATAMA,
    CAN_HAVE_DAGESH, CAN_HAVE_SIN, LETTERS
)


def reconstruct_text(
    input_ids: np.ndarray,
    vowel_preds: np.ndarray,
    dagesh_preds: np.ndarray,
    sin_preds: np.ndarray,
    stress_preds: np.ndarray,
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
    result = []
    
    # Get special token strings (for tokenizers library)
    # SEP and PAD tokens are typically [SEP] and [PAD]
    
    # Skip [CLS] (position 0) and stop at [SEP] or [PAD]
    for i in range(1, len(input_ids)):
        token_id = int(input_ids[i])
        
        # Decode token
        char = tokenizer.decode([token_id])
        
        # Stop at special tokens
        if char in ['[SEP]', '[PAD]', '']:
            break
        result.append(char)
        
        # Only add nikud marks for Hebrew letters
        if char not in LETTERS:
            continue
        
        diacritics = []
        
        # Add vowel (if not VOWEL_NONE)
        vowel_id = int(vowel_preds[i])
        if vowel_id > 0:  # 0 is VOWEL_NONE
            vowel_char = ID_TO_VOWEL.get(vowel_id)
            if vowel_char:
                diacritics.append(vowel_char)
        
        # Add dagesh if predicted and character supports it
        if int(dagesh_preds[i]) == 1 and char in CAN_HAVE_DAGESH:
            diacritics.append(DAGESH)
        
        # Add sin if predicted and character supports it
        if int(sin_preds[i]) == 1 and char in CAN_HAVE_SIN:
            diacritics.append(S_SIN)
        
        # Add stress if predicted
        if int(stress_preds[i]) == 1:
            diacritics.append(STRESS_HATAMA)
        
        # Sort diacritics for canonical order
        diacritics.sort()
        result.extend(diacritics)
    
    # Combine and normalize
    text = ''.join(result)
    text = unicodedata.normalize('NFD', text)
    text = unicodedata.normalize('NFC', text)
    
    return text

