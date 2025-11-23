"""
Decoding logic to convert model outputs to nikud text.
"""

import numpy as np

from .predict import reconstruct_text


def decode_predictions(
    outputs: tuple,
    input_ids: np.ndarray,
    tokenizer
) -> str:
    """
    Decode model outputs into Hebrew text with nikud.
    
    Args:
        outputs: Tuple of (vowel_logits, dagesh_logits, sin_logits, stress_logits)
        input_ids: Token IDs [batch_size, seq_len]
        tokenizer: Tokenizer for decoding
        
    Returns:
        Text with predicted nikud marks
    """
    vowel_logits = outputs[0]   # [batch_size, seq_len, 6]
    dagesh_logits = outputs[1]  # [batch_size, seq_len]
    sin_logits = outputs[2]     # [batch_size, seq_len]
    stress_logits = outputs[3]  # [batch_size, seq_len]
    
    # Convert logits to predictions
    vowel_preds = np.argmax(vowel_logits[0], axis=-1)  # [seq_len]
    dagesh_preds = sigmoid(dagesh_logits[0]) > 0.5
    sin_preds = sigmoid(sin_logits[0]) > 0.5
    stress_preds = sigmoid(stress_logits[0]) > 0.5
    
    # Reconstruct text with nikud
    result = reconstruct_text(
        input_ids[0],
        vowel_preds,
        dagesh_preds.astype(np.int32),
        sin_preds.astype(np.int32),
        stress_preds.astype(np.int32),
        tokenizer
    )
    
    return result


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

