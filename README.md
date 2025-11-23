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

# Upload model to Hugging Face

```console
uv pip install huggingface_hub
git config --global credential.helper store # Allow clone private repo from HF
# Get token from https://huggingface.co/settings/tokens
uv run hf auth login --token "token" --add-to-git-credential #
uv run hf upload --repo-type model renikud-v2 ./checkpoints/checkpoint-10000
```

# Download model from Hugging Face

```console
uv run hf download --repo-type model thewh1teagle/renikud-v2 --local-dir ./checkpoints/latest
```

# Huggingface Model Card

See [model card](https://huggingface.co/thewh1teagle/renikud-v2) for more details.