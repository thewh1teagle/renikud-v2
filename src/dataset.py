"""
Dataset preparation for Hebrew nikud prediction.

This module handles:
- Extracting nikud marks from Hebrew text
- Creating input/label pairs for training
- Encoding labels for vowels, dagesh, sin, and stress marks
"""

import unicodedata
from typing import List, Tuple, Dict
import torch
from normalize import normalize
from constants import (
    A_PATAH, E_TSERE, I_HIRIK, O_HOLAM, U_QUBUT,
    DAGESH, S_SIN, STRESS_HATAMA,
    CAN_HAVE_DAGESH, CAN_HAVE_SIN, LETTERS
)


# Label encoding for vowels
VOWEL_TO_ID = {
    None: 0,  # No vowel
    A_PATAH: 1,
    E_TSERE: 2,
    I_HIRIK: 3,
    O_HOLAM: 4,
    U_QUBUT: 5,
}

ID_TO_VOWEL = {v: k for k, v in VOWEL_TO_ID.items()}


def extract_nikud_labels(nikud_text: str) -> Tuple[str, List[Dict[str, int]]]:
    """
    Extract nikud labels from Hebrew text.
    
    Args:
        nikud_text: Hebrew text with nikud marks
        
    Returns:
        Tuple of (plain_text, labels) where:
        - plain_text: Hebrew text without nikud marks
        - labels: List of dicts with keys 'vowel', 'dagesh', 'sin', 'stress' for each character
    """
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
            'vowel': 0,  # No vowel by default
            'dagesh': 0,  # No dagesh by default
            'sin': 0,    # No sin by default
            'stress': 0  # No stress by default
        }
        
        # Look ahead for diacritics
        j = i + 1
        while j < len(nikud_text) and unicodedata.category(nikud_text[j]) in ['Mn', 'Me']:
            diacritic = nikud_text[j]
            
            # Check for vowel
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


def prepare_training_data(nikud_text: str, tokenizer) -> Dict[str, torch.Tensor]:
    """
    Prepare training data from nikud'd Hebrew text.
    
    Args:
        nikud_text: Hebrew text with nikud marks
        tokenizer: HuggingFace tokenizer for the model
        
    Returns:
        Dictionary with input_ids, attention_mask, and label tensors
    """
    plain_text, labels = extract_nikud_labels(nikud_text)
    
    # Tokenize the plain text
    encoding = tokenizer(
        plain_text,
        return_tensors='pt',
        padding=False,
        truncation=False,
        add_special_tokens=True
    )
    
    # The tokenizer is character-level, so we need to align labels with tokens
    # Get token ids (excluding special tokens for label alignment)
    input_ids = encoding['input_ids'][0]
    
    # Create label tensors
    # We need to handle special tokens [CLS] and [SEP]
    # Labels for special tokens should be -100 (ignored in loss)
    num_tokens = len(input_ids)
    
    vowel_labels = torch.full((num_tokens,), -100, dtype=torch.long)
    dagesh_labels = torch.full((num_tokens,), -100, dtype=torch.long)
    sin_labels = torch.full((num_tokens,), -100, dtype=torch.long)
    stress_labels = torch.full((num_tokens,), -100, dtype=torch.long)
    
    # Fill in labels for actual characters (skip [CLS] at position 0)
    # Assuming character-level tokenization: token i corresponds to character i-1
    for i, label in enumerate(labels):
        token_idx = i + 1  # +1 to account for [CLS] token
        if token_idx < num_tokens - 1:  # -1 to avoid [SEP]
            vowel_labels[token_idx] = label['vowel']
            dagesh_labels[token_idx] = label['dagesh']
            sin_labels[token_idx] = label['sin']
            stress_labels[token_idx] = label['stress']
    
    return {
        'input_ids': encoding['input_ids'][0],
        'attention_mask': encoding['attention_mask'][0],
        'vowel_labels': vowel_labels,
        'dagesh_labels': dagesh_labels,
        'sin_labels': sin_labels,
        'stress_labels': stress_labels,
        'plain_text': plain_text,
        'original_text': nikud_text,  # Already in NFD format
    }


class NikudDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Hebrew nikud prediction."""
    
    def __init__(self, texts: List[str], tokenizer):
        """
        Args:
            texts: List of Hebrew texts with nikud marks
            tokenizer: HuggingFace tokenizer
        """
        self.data = [prepare_training_data(text, tokenizer) for text in texts]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_dataset_from_file(file_path: str) -> List[str]:
    """
    Load Hebrew texts from a file.
    
    Args:
        file_path: Path to text file (one text per line)
        
    Returns:
        List of texts with nikud marks
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                texts.append(line)
    return texts


def split_dataset(texts: List[str], eval_max_lines: int, seed: int = 42) -> tuple:
    """
    Split dataset into train and eval sets.
    
    Args:
        texts: List of texts with nikud marks
        eval_max_lines: Maximum number of lines to use for evaluation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, eval_texts)
    """
    import random
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle texts
    shuffled_texts = texts.copy()
    random.shuffle(shuffled_texts)
    
    # Use minimum of eval_max_lines and total texts
    eval_size = min(eval_max_lines, len(shuffled_texts))
    
    # Split
    eval_texts = shuffled_texts[:eval_size]
    train_texts = shuffled_texts[eval_size:]
    
    return train_texts, eval_texts


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of data dictionaries from NikudDataset
        
    Returns:
        Dictionary with batched and padded tensors
    """
    # Find max length in batch
    max_len = max(item['input_ids'].shape[0] for item in batch)
    
    # Initialize batched tensors
    batch_size = len(batch)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    vowel_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    dagesh_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    sin_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    stress_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    plain_texts = []
    original_texts = []
    
    # Fill in the batch
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        vowel_labels[i, :seq_len] = item['vowel_labels']
        dagesh_labels[i, :seq_len] = item['dagesh_labels']
        sin_labels[i, :seq_len] = item['sin_labels']
        stress_labels[i, :seq_len] = item['stress_labels']
        
        plain_texts.append(item['plain_text'])
        original_texts.append(item['original_text'])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'vowel_labels': vowel_labels,
        'dagesh_labels': dagesh_labels,
        'sin_labels': sin_labels,
        'stress_labels': stress_labels,
        'plain_text': plain_texts,
        'original_text': original_texts,
    }

