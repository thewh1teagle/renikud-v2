# Renikud v2

Rethinking Hebrew nikud with a new approach to models and G2P

## Training

Train the model on Hebrew text with nikud:

```bash
uv run src/prepare_data.py
uv run python src/train.py --epochs 100 --lr 1e-4
```

With wandb:
```bash
export WANDB_API_KEY="api key" # https://wandb.ai/authorize
export WANDB_PROJECT="renikud-v2"
uv run src/train.py --wandb-mode online ... # see src/train.py for all options
```

## Inference

Add nikud to plain Hebrew text:

```bash
# Single text
uv run python src/inference.py --checkpoint checkpoints/best_model.pt --text "האיש שלא רצה"

# From file
uv run python src/inference.py --checkpoint checkpoints/best_model.pt --file input.txt
```

