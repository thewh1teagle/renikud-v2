# Renikud v2

Rethinking Hebrew nikud with a new approach to models and G2P

## Training

Train the model on Hebrew text with nikud:

```bash
uv run src/prepare_data.py
uv run python src/train.py --epochs 100 --lr 1e-4
```

## Inference

Add nikud to plain Hebrew text:

```bash
# Single text
uv run python src/inference.py --checkpoint checkpoints/best_model.pt --text "האיש שלא רצה"

# From file
uv run python src/inference.py --checkpoint checkpoints/best_model.pt --file input.txt
```

