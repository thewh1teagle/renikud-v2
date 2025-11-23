#!/usr/bin/env python3
"""
Export Hebrew Nikud model from safetensors to ONNX format.
"""

import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.renikud_onnx.model import HebrewNikudModel


def export_to_onnx(checkpoint_path: str, output_path: str):
    """
    Export model from safetensors checkpoint to ONNX.
    
    Args:
        checkpoint_path: Path to checkpoint (directory or .safetensors file)
        output_path: Path to save ONNX model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model architecture
    model = HebrewNikudModel()
    
    # Load checkpoint weights
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / 'model.safetensors'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(load_file(checkpoint_path, device='cpu'))
    model.eval()
    
    print("Model loaded successfully")
    
    # Create dummy inputs for ONNX export
    batch_size = 1
    seq_length = 128
    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['vowel_logits', 'dagesh_logits', 'sin_logits', 'stress_logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'vowel_logits': {0: 'batch_size', 1: 'sequence_length'},
            'dagesh_logits': {0: 'batch_size', 1: 'sequence_length'},
            'sin_logits': {0: 'batch_size', 1: 'sequence_length'},
            'stress_logits': {0: 'batch_size', 1: 'sequence_length'},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    
    print(f"✓ ONNX model exported successfully to: {output_path}")
    
    # Export int8 quantized version
    output_path_obj = Path(output_path)
    int8_path = output_path_obj.parent / f"{output_path_obj.stem}_int8{output_path_obj.suffix}"
    
    print(f"\nQuantizing to int8: {int8_path}")
    try:
        quantize_dynamic(
            model_input=output_path,
            model_output=str(int8_path),
            weight_type=QuantType.QInt8
        )
        
        # Compare file sizes
        original_size = Path(output_path).stat().st_size
        int8_size = int8_path.stat().st_size
        
        print(f"✓ Int8 model exported successfully to: {int8_path}")
    except Exception as e:
        print(f"⚠ Int8 quantization failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Export Hebrew Nikud model to ONNX format'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (directory or .safetensors file)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='model.onnx',
        help='Output path for ONNX model (default: model.onnx)'
    )
    
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
