# Renikud v2

Rethinking Hebrew nikud with a new approach to models and G2P

## Training

Train the model on Hebrew text with nikud:

```bash
uv run python src/train.py --epochs 100 --lr 1e-4
```

Options:
- `--text`: Training text with nikud (default: "הַאִיש שֵלֹא רַצַה לִהיוֹת אַגַדַה")
- `--epochs`: Number of training epochs (default: 500)
- `--lr`: Learning rate (default: 1e-4)
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)

## Inference

Add nikud to plain Hebrew text:

```bash
# Single text
uv run python src/inference.py --checkpoint checkpoints/best_model.pt --text "האיש שלא רצה"

# Interactive mode
uv run python src/inference.py --checkpoint checkpoints/best_model.pt

# From file
uv run python src/inference.py --checkpoint checkpoints/best_model.pt --file input.txt
```


Prepare for training:

```console
wget https://huggingface.co/datasets/thewh1teagle/phonikud-data/resolve/main/knesset_nikud_v6.txt.7z
sudo apt install p7zip-full -y
7z x knesset_nikud_v6.txt.7z
```