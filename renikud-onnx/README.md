# renikud-onnx

Fast ONNX inference for Hebrew Nikud prediction.

## Installation

```bash
pip install renikud-onnx
```

## Usage

```python
from renikud_onnx import Renikud

model = Renikud('nikud_model.onnx')
text = "הוא רצה את זה גם, אבל היא רצה מהר והקדימה אותו!"
nikud_text = model.add_diacritics(text)

print(nikud_text)
# הוּא רַצַה אֵת זֵה גַם, אַבַל הִיא רַצַה מַהֵר והִקדִ֫ימַה אוֹתוֹ!
```

## Export Model to ONNX

```bash
python export.py --checkpoint /path/to/checkpoint --output model.onnx
```

## Development

```bash
# Install dependencies
uv sync --dev

# Export a model
uv run --dev python export.py --checkpoint /path/to/checkpoint --output model.onnx

# Run example
uv run python examples/usage.py
```
