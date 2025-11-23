"""
Encoding Hebrew text with nikud into training labels.

This is the inverse of decode.py - it takes text with nikud marks
and extracts the labels for training.
"""

import unicodedata
from typing import Tuple, List, Dict


def extract_nikud_labels(nikud_text: str) -> Tuple[str, List[Dict[str, int]]]:
    """
    Extract nikud labels from Hebrew text.
    
    Args:
        nikud_text: Hebrew text with nikud marks
        
    Returns:
        Tuple of (plain_text, labels) where:
        - plain_text: Hebrew text without nikud marks
        - labels: List of dicts with keys 'vowel' (0-5), 'dagesh', 'sin', 'stress' (0/1)
    """
    # Import here to avoid circular dependencies
    from normalize import normalize
    from constants import (
        LETTERS, DAGESH, S_SIN, STRESS_HATAMA,
        CAN_HAVE_DAGESH, CAN_HAVE_SIN
    )
    from dataset import VOWEL_TO_ID, VOWEL_NONE
    
    # Normalize the text first
    nikud_text = normalize(nikud_text)
    
    # Decompose to separate characters and diacritics
    nikud_text = unicodedata.normalize('NFD', nikud_text)
    
    plain_chars = []
    labels = []
    
    i = 0
    while i < len(nikud_text):
        char = nikud_text[i]
        
        # Handle non-Hebrew letters (spaces, punctuation, etc.)
        if char not in LETTERS:
            plain_chars.append(char)
            # Mark as non-classifiable with -100 (will be ignored in loss)
            labels.append({
                'vowel': -100,
                'dagesh': -100,
                'sin': -100,
                'stress': -100
            })
            i += 1
            continue
            
        plain_chars.append(char)
        
        # Initialize labels for this Hebrew letter
        label = {
            'vowel': VOWEL_NONE,  # No vowel by default
            'dagesh': 0,          # No dagesh by default
            'sin': 0,             # No sin by default
            'stress': 0           # No stress by default
        }
        
        # Look ahead for diacritics
        j = i + 1
        while j < len(nikud_text) and unicodedata.category(nikud_text[j]) in ['Mn', 'Me']:
            diacritic = nikud_text[j]
            
            # Check for vowel (only one vowel per character)
            if diacritic in VOWEL_TO_ID:
                label['vowel'] = VOWEL_TO_ID[diacritic]
            # Check for dagesh (only valid on specific letters)
            elif diacritic == DAGESH and char in CAN_HAVE_DAGESH:
                label['dagesh'] = 1
            # Check for sin (only valid on shin)
            elif diacritic == S_SIN and char in CAN_HAVE_SIN:
                label['sin'] = 1
            # Check for stress
            elif diacritic == STRESS_HATAMA:
                label['stress'] = 1
                
            j += 1
        
        labels.append(label)
        i = j
    
    plain_text = ''.join(plain_chars)
    return plain_text, labels

