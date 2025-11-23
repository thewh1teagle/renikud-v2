"""
Hebrew Nikud BERT Model for ONNX export.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class HebrewNikudModel(nn.Module):
    """
    BERT-based model for Hebrew nikud prediction.
    
    Outputs:
    - vowel_logits: 6 classes (none, patah, tsere, hirik, holam, qubut)
    - dagesh_logits: binary (dagesh presence)
    - sin_logits: binary (sin/shin dot)
    - stress_logits: binary (stress mark)
    """
    
    def __init__(self, model_name: str = 'dicta-il/dictabert-large-char', dropout: float = 0.1):
        super().__init__()
        
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification heads
        self.vowel_classifier = nn.Linear(self.hidden_size, 6)
        self.dagesh_classifier = nn.Linear(self.hidden_size, 1)
        self.sin_classifier = nn.Linear(self.hidden_size, 1)
        self.stress_classifier = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass for inference.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (vowel_logits, dagesh_logits, sin_logits, stress_logits)
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Get predictions from each head
        vowel_logits = self.vowel_classifier(sequence_output)  # [batch_size, seq_len, 6]
        dagesh_logits = self.dagesh_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        sin_logits = self.sin_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        stress_logits = self.stress_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        
        return vowel_logits, dagesh_logits, sin_logits, stress_logits

