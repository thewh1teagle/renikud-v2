"""
Hebrew Nikud BERT Model with hybrid classification heads.

This module defines a custom BERT model for predicting Hebrew nikud marks:
- Vowel prediction: 1 multi-class head with 6 classes (none + 5 vowels) - mutually exclusive
- Other marks: 3 binary heads (dagesh, sin, stress) - independent
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Tuple, Dict


class HebrewNikudModel(nn.Module):
    """
    BERT-based model for Hebrew nikud prediction with hybrid classification.
    
    Architecture:
    - Vowel classifier: 6 classes (none, patah, tsere, hirik, holam, qubut) - uses CrossEntropy
    - Dagesh classifier: binary - uses BCE
    - Sin classifier: binary - uses BCE
    - Stress classifier: binary - uses BCE
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
        self.dagesh_classifier = nn.Linear(self.hidden_size, 1)  # binary (sigmoid output)
        self.sin_classifier = nn.Linear(self.hidden_size, 1)     # binary (sigmoid output)
        self.stress_classifier = nn.Linear(self.hidden_size, 1)  # binary (sigmoid output)
        
        # Loss functions
        # Using label smoothing to prevent overconfidence and help with class imbalance
        self.vowel_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vowel_labels: Optional[torch.Tensor] = None,
        dagesh_labels: Optional[torch.Tensor] = None,
        sin_labels: Optional[torch.Tensor] = None,
        stress_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            vowel_labels: Vowel class labels [batch_size, seq_len] (0-5, -100 for ignore)
            dagesh_labels: Dagesh binary labels [batch_size, seq_len] (0/1, -100 for ignore)
            sin_labels: Sin binary labels [batch_size, seq_len] (0/1, -100 for ignore)
            stress_labels: Stress binary labels [batch_size, seq_len] (0/1, -100 for ignore)
            
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
        dagesh_logits = self.dagesh_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        sin_logits = self.sin_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        stress_logits = self.stress_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        
        result = {
            'vowel_logits': vowel_logits,
            'dagesh_logits': dagesh_logits,
            'sin_logits': sin_logits,
            'stress_logits': stress_logits,
        }
        
        # Calculate losses if labels are provided
        if vowel_labels is not None:
            # Vowel loss (multi-class)
            vowel_loss = self.vowel_loss_fn(
                vowel_logits.view(-1, 6),
                vowel_labels.view(-1)
            )
            
            # Binary losses with masking
            dagesh_mask = (dagesh_labels != -100).float()
            sin_mask = (sin_labels != -100).float()
            stress_mask = (stress_labels != -100).float()
            
            # Replace -100 with 0 for loss computation
            dagesh_labels_masked = dagesh_labels.clone().float()
            dagesh_labels_masked[dagesh_labels == -100] = 0.0
            
            sin_labels_masked = sin_labels.clone().float()
            sin_labels_masked[sin_labels == -100] = 0.0
            
            stress_labels_masked = stress_labels.clone().float()
            stress_labels_masked[stress_labels == -100] = 0.0
            
            # Compute binary losses
            dagesh_loss_unreduced = self.binary_loss_fn(dagesh_logits, dagesh_labels_masked)
            dagesh_loss = (dagesh_loss_unreduced * dagesh_mask).sum() / (dagesh_mask.sum() + 1e-8)
            
            sin_loss_unreduced = self.binary_loss_fn(sin_logits, sin_labels_masked)
            sin_loss = (sin_loss_unreduced * sin_mask).sum() / (sin_mask.sum() + 1e-8)
            
            stress_loss_unreduced = self.binary_loss_fn(stress_logits, stress_labels_masked)
            stress_loss = (stress_loss_unreduced * stress_mask).sum() / (stress_mask.sum() + 1e-8)
            
            # Combined loss
            total_loss = vowel_loss + dagesh_loss + sin_loss + stress_loss
            
            result.update({
                'loss': total_loss,
                'vowel_loss': vowel_loss,
                'dagesh_loss': dagesh_loss,
                'sin_loss': sin_loss,
                'stress_loss': stress_loss,
            })
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for nikud marks.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with predictions:
            - vowel: class predictions [batch_size, seq_len] (0-5)
            - dagesh: binary predictions [batch_size, seq_len] (0/1)
            - sin: binary predictions [batch_size, seq_len] (0/1)
            - stress: binary predictions [batch_size, seq_len] (0/1)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            # Vowel: argmax over classes
            vowel_preds = torch.argmax(outputs['vowel_logits'], dim=-1)
            
            # Binary: sigmoid + threshold
            dagesh_preds = (torch.sigmoid(outputs['dagesh_logits']) > 0.5).long()
            sin_preds = (torch.sigmoid(outputs['sin_logits']) > 0.5).long()
            stress_preds = (torch.sigmoid(outputs['stress_logits']) > 0.5).long()
            
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


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
