"""
Hebrew Nikud BERT Model with 4 classification heads.

This module defines a custom BERT model for predicting Hebrew nikud marks:
- Vowel prediction (6 classes)
- Dagesh prediction (binary)
- Sin prediction (binary)
- Stress prediction (binary)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Tuple, Dict
from constants import CAN_HAVE_DAGESH, CAN_HAVE_SIN


class HebrewNikudModel(nn.Module):
    """
    BERT-based model for Hebrew nikud prediction with 4 separate classification heads.
    """
    
    def __init__(self, model_name: str = 'dicta-il/dictabert-large-char', dropout: float = 0.1):
        """
        Args:
            model_name: Name of the pretrained BERT model to use
            dropout: Dropout probability for classification heads
        """
        super().__init__()
        
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification heads
        self.vowel_classifier = nn.Linear(self.hidden_size, 6)  # 6 classes: none + 5 vowels
        self.dagesh_classifier = nn.Linear(self.hidden_size, 2)  # binary
        self.sin_classifier = nn.Linear(self.hidden_size, 2)     # binary
        self.stress_classifier = nn.Linear(self.hidden_size, 2)  # binary
        
        # Loss functions
        self.vowel_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.dagesh_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.sin_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.stress_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vowel_labels: Optional[torch.Tensor] = None,
        dagesh_labels: Optional[torch.Tensor] = None,
        sin_labels: Optional[torch.Tensor] = None,
        stress_labels: Optional[torch.Tensor] = None,
        tokenizer: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            vowel_labels: Vowel labels [batch_size, seq_len]
            dagesh_labels: Dagesh labels [batch_size, seq_len]
            sin_labels: Sin labels [batch_size, seq_len]
            stress_labels: Stress labels [batch_size, seq_len]
            tokenizer: Tokenizer for decoding tokens (optional, for masking)
            
        Returns:
            Dictionary with logits and optionally loss
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Get predictions from each head
        vowel_logits = self.vowel_classifier(sequence_output)  # [batch_size, seq_len, 6]
        dagesh_logits = self.dagesh_classifier(sequence_output)  # [batch_size, seq_len, 2]
        sin_logits = self.sin_classifier(sequence_output)       # [batch_size, seq_len, 2]
        stress_logits = self.stress_classifier(sequence_output)  # [batch_size, seq_len, 2]
        
        # Apply masking for invalid predictions (if tokenizer is provided)
        if tokenizer is not None:
            dagesh_logits = self._mask_invalid_dagesh(input_ids, dagesh_logits, tokenizer)
            sin_logits = self._mask_invalid_sin(input_ids, sin_logits, tokenizer)
        
        result = {
            'vowel_logits': vowel_logits,
            'dagesh_logits': dagesh_logits,
            'sin_logits': sin_logits,
            'stress_logits': stress_logits,
        }
        
        # Calculate losses if labels are provided
        if vowel_labels is not None:
            # Flatten for loss calculation
            batch_size, seq_len = input_ids.shape
            
            vowel_loss = self.vowel_loss_fn(
                vowel_logits.view(-1, 6),
                vowel_labels.view(-1)
            )
            dagesh_loss = self.dagesh_loss_fn(
                dagesh_logits.view(-1, 2),
                dagesh_labels.view(-1)
            )
            sin_loss = self.sin_loss_fn(
                sin_logits.view(-1, 2),
                sin_labels.view(-1)
            )
            stress_loss = self.stress_loss_fn(
                stress_logits.view(-1, 2),
                stress_labels.view(-1)
            )
            
            # Combined loss (can adjust weights)
            total_loss = vowel_loss + dagesh_loss + sin_loss + stress_loss
            
            result.update({
                'loss': total_loss,
                'vowel_loss': vowel_loss,
                'dagesh_loss': dagesh_loss,
                'sin_loss': sin_loss,
                'stress_loss': stress_loss,
            })
        
        return result
    
    def _mask_invalid_dagesh(
        self,
        input_ids: torch.Tensor,
        dagesh_logits: torch.Tensor,
        tokenizer
    ) -> torch.Tensor:
        """
        Mask dagesh predictions for characters that cannot have dagesh.
        
        Sets logit for positive class to -inf for invalid positions.
        """
        # Get the actual characters from input_ids
        for batch_idx in range(input_ids.shape[0]):
            for token_idx in range(input_ids.shape[1]):
                token_id = input_ids[batch_idx, token_idx].item()
                token_char = tokenizer.decode([token_id])
                
                # If not in CAN_HAVE_DAGESH, mask the positive class
                if token_char not in CAN_HAVE_DAGESH:
                    dagesh_logits[batch_idx, token_idx, 1] = float('-inf')
        
        return dagesh_logits
    
    def _mask_invalid_sin(
        self,
        input_ids: torch.Tensor,
        sin_logits: torch.Tensor,
        tokenizer
    ) -> torch.Tensor:
        """
        Mask sin predictions for characters that cannot have sin.
        
        Sets logit for positive class to -inf for invalid positions.
        """
        # Get the actual characters from input_ids
        for batch_idx in range(input_ids.shape[0]):
            for token_idx in range(input_ids.shape[1]):
                token_id = input_ids[batch_idx, token_idx].item()
                token_char = tokenizer.decode([token_id])
                
                # If not shin, mask the positive class
                if token_char not in CAN_HAVE_SIN:
                    sin_logits[batch_idx, token_idx, 1] = float('-inf')
        
        return sin_logits
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for nikud marks.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            tokenizer: Tokenizer for decoding
            
        Returns:
            Dictionary with predicted labels
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, tokenizer=tokenizer)
            
            # Get predicted classes
            vowel_preds = torch.argmax(outputs['vowel_logits'], dim=-1)
            dagesh_preds = torch.argmax(outputs['dagesh_logits'], dim=-1)
            sin_preds = torch.argmax(outputs['sin_logits'], dim=-1)
            stress_preds = torch.argmax(outputs['stress_logits'], dim=-1)
            
            return {
                'vowel': vowel_preds,
                'dagesh': dagesh_preds,
                'sin': sin_preds,
                'stress': stress_preds,
            }


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[HebrewNikudModel, object]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char')
    
    # Load model
    model = HebrewNikudModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer

