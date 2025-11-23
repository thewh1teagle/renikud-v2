"""
renikud-onnx: Fast ONNX inference for Hebrew Nikud prediction.
"""

import re
import onnxruntime
import numpy as np
from pathlib import Path
from typing import Union
from tokenizers import Tokenizer

from .decode import decode_predictions


class Renikud:
    """Hebrew Nikud predictor using ONNX runtime."""
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the predictor with an ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.session = onnxruntime.InferenceSession(str(model_path))
        self.tokenizer = Tokenizer.from_pretrained('dicta-il/dictabert-large-char')
    
    def add_diacritics(self, text: str) -> str:
        """
        Add nikud (diacritical marks) to Hebrew text.
        
        Args:
            text: Plain Hebrew text without nikud
            
        Returns:
            Text with predicted nikud marks
        """
        # Remove diacritics from the text
        text = re.sub(r'[\u0590-\u05CF]', '', text)
        
        # Tokenize
        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        
        # Run inference
        outputs = self.session.run(
            None,
            {'input_ids': input_ids, 'attention_mask': attention_mask}
        )
        
        # Decode predictions
        return decode_predictions(outputs, input_ids, self.tokenizer)


__all__ = ['Renikud']
