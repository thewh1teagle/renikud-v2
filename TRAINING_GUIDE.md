# Hebrew Nikud Training Guide

This guide explains how to use the refactored training pipeline.

## Overview

The training pipeline has been refactored into modular components:

- `src/config.py` - Configuration management with argparse
- `src/train_loop.py` - Main training orchestration
- `src/evaluate.py` - Evaluation with WER/CER metrics
- `src/dataset.py` - Dataset loading, splitting, and batching
- `src/model.py` - Model definition and utilities
- `src/main.py` - Entry point

## Quick Start

### Basic Training

```bash
uv run python src/main.py
```

This will train with default settings:
- Dataset: `data/train.txt`
- Batch size: 8
- Learning rate: 1e-4
- Max epochs: 10
- Eval ratio: 0.2 (20% for validation)
- Wandb mode: disabled
- Random seed: 42

### Custom Training

```bash
uv run python src/main.py \
  --train-file data/train.txt \
  --batch-size 16 \
  --lr 2e-4 \
  --max-epochs 20 \
  --eval-max-lines 100 \
  --wandb-mode online \
  --wandb-project my-hebrew-nikud \
  --checkpoint-dir checkpoints
```

## Configuration Options

### Dataset
- `--train-file`: Path to training data file (default: `data/train.txt`)
- `--eval-max-lines`: Maximum number of lines for evaluation (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

### Training
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--max-epochs`: Maximum training epochs (default: 10)
- `--max-grad-norm`: Maximum gradient norm for clipping (default: 1.0)

### Model
- `--model-name`: Pretrained model name (default: `dicta-il/dictabert-large-char`)
- `--dropout`: Dropout rate (default: 0.1)

### Checkpoints
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints`)

### Device
- `--device`: Device to train on - `cpu`, `cuda`, or `mps` (default: auto-detect)

### Wandb
- `--wandb-mode`: Wandb logging mode - `online`, `offline`, or `disabled` (default: `disabled`)
- `--wandb-project`: Wandb project name (default: `hebrew-nikud`)
- `--wandb-run-name`: Wandb run name (default: None)

## Features

### Training Progress
- **tqdm progress bars** show real-time training progress with loss values
- Epoch-by-epoch summary with all metrics

### Model Information
Before training starts, the script prints:
- Total parameters
- Trainable parameters
- Approximate model size in MB

### Metrics Tracked

**Training metrics:**
- Total loss
- Vowel loss
- Dagesh loss
- Sin loss
- Stress loss

**Evaluation metrics:**
- Total loss
- **WER (Word Error Rate)** - word-level accuracy
- **CER (Character Error Rate)** - character-level accuracy
- Vowel accuracy
- Dagesh accuracy
- Sin accuracy
- Stress accuracy

### Checkpointing
- Best model is saved based on evaluation loss
- Final model is saved at the end of training
- Checkpoints saved as `best_model.pt` and `final_model.pt`

### Reproducibility
- Random seed controls all randomness (Python, NumPy, PyTorch)
- Dataset split is deterministic with the same seed

### Wandb Integration
All metrics are logged to Wandb:
- `train_loss`, `train_vowel_loss`, etc.
- `eval_loss`, `eval_wer`, `eval_cer`, etc.
- Model parameter counts
- Learning rate

## Data Format

The training file should contain one text sample per line with nikud marks:

```
הַאִיש שֵלֹא רַצַה לִהיוֹת אַגַדַה
הַיֹום הוּא יוֹם יָפֶה
...
```

## Example Workflow

1. **Prepare your data**: Ensure `data/train.txt` contains your training texts

2. **Start training**:
   ```bash
   uv run python src/main.py --max-epochs 20 --wandb-mode online
   ```

3. **Monitor progress**: Watch the terminal for tqdm progress bars and epoch summaries

4. **Check results**: 
   - Best model: `checkpoints/best_model.pt`
   - Final model: `checkpoints/final_model.pt`
   - Wandb dashboard: View detailed metrics and comparisons

5. **Use the model**: Load the checkpoint with `src/inference.py`

## Tips

- Start with `--wandb-mode disabled` for no tracking (fastest)
- Use `--wandb-mode offline` to log locally without Wandb account
- Use `--wandb-mode online` when ready to log to Wandb servers
- Adjust `--batch-size` based on your GPU memory
- Increase `--max-epochs` for better convergence
- Use `--seed` to reproduce results exactly

## Next Steps

After training, you can:
- Use `src/inference.py` to add nikud to new texts
- Adjust hyperparameters and retrain
- Compare different runs in Wandb
- Fine-tune on specific domains

