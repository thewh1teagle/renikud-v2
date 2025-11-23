"""
Dataset preparation for Hebrew nikud prediction.

This module handles:
- Creating input/label pairs for training
- Hybrid encoding: vowels as multi-class (0-5), others as binary
"""

from typing import List
import torch
from tqdm import tqdm
import hashlib
import pickle
import os
from pathlib import Path
from encode import extract_nikud_labels
from constants import A_PATAH, E_TSERE, I_HIRIK, O_HOLAM, U_QUBUT


# Vowel encoding (multi-class)
VOWEL_NONE = 0
VOWEL_PATAH = 1
VOWEL_TSERE = 2
VOWEL_HIRIK = 3
VOWEL_HOLAM = 4
VOWEL_QUBUT = 5

VOWEL_TO_ID = {
    None: VOWEL_NONE,
    A_PATAH: VOWEL_PATAH,
    E_TSERE: VOWEL_TSERE,
    I_HIRIK: VOWEL_HIRIK,
    O_HOLAM: VOWEL_HOLAM,
    U_QUBUT: VOWEL_QUBUT,
}

ID_TO_VOWEL = {v: k for k, v in VOWEL_TO_ID.items()}


def _load_or_process_dataset(texts: List[str], tokenizer, cache_dir: str, use_cache: bool) -> List[dict]:
    """Load dataset from cache or process and cache it."""
    # If caching disabled, just process
    if not use_cache:
        print(f"Processing {len(texts)} texts (caching disabled)...")
        return [
            prepare_training_data(text, tokenizer) 
            for text in tqdm(texts, desc="Preparing dataset", unit="texts")
        ]
    
    # Caching enabled
    Path(cache_dir).mkdir(exist_ok=True)
    
    # Generate cache key
    data_str = "".join(texts) + str(tokenizer.vocab_size)
    data_hash = hashlib.md5(data_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"dataset_{data_hash}.pkl")
    
    # Try loading from cache
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} cached samples")
        return data
    
    # Process and cache
    print(f"Processing {len(texts)} texts...")
    data = [
        prepare_training_data(text, tokenizer) 
        for text in tqdm(texts, desc="Preparing dataset", unit="texts")
    ]
    
    print(f"Saving dataset to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Cached {len(data)} samples")
    
    return data


def prepare_training_data(nikud_text: str, tokenizer) -> dict:
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
    input_ids = encoding['input_ids'][0]
    num_tokens = len(input_ids)
    
    # Create label tensors
    # Labels for special tokens should be -100 (ignored in loss)
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
    
    def __init__(self, texts: List[str], tokenizer, cache_dir: str = ".dataset_cache", use_cache: bool = True):
        """
        Args:
            texts: List of Hebrew texts with nikud marks
            tokenizer: HuggingFace tokenizer
            cache_dir: Directory to cache processed datasets
            use_cache: Whether to cache the processed dataset
        """
        self.data = _load_or_process_dataset(texts, tokenizer, cache_dir, use_cache)
    
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


def collate_fn(batch: List[dict]) -> dict:
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
