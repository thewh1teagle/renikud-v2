# Hebrew Nikud Model Documentation

## Purpose and Overview

This model automatically adds nikud (vowel marks and diacritics) to Hebrew text. Hebrew is typically written without vowels, which can make it harder to read and can create ambiguity. This model predicts and adds the appropriate nikud marks to make the text easier to read.

**Example:**
- Input: `האיש רצה`
- Output: `הָאִישׁ רָצָה`

## Architecture

### Base Model
The model is built on **DictaBERT Large** (`dicta-il/dictabert-large-char`), a character-level BERT model pre-trained on Hebrew text. Character-level tokenization is crucial for Hebrew nikud prediction since we need to predict marks for individual characters.

### Classification Heads

The model uses a hybrid approach with **4 separate classification heads**:

1. **Vowel Classifier** (6-class, mutually exclusive)
   - Classes: 0=none, 1=patah (ַ), 2=tsere (ֵ), 3=hirik (ִ), 4=holam (ֹ), 5=qubut (ֻ)
   - Uses `CrossEntropyLoss` with label smoothing (0.1)
   - A letter can have at most one vowel

2. **Dagesh Classifier** (binary)
   - Predicts whether the dagesh mark (ּ) appears
   - Uses `BCEWithLogitsLoss`
   - Only applies to certain letters: ב, כ, פ, ו

3. **Sin Classifier** (binary)
   - Predicts the sin dot (שׂ) on the letter ש
   - Uses `BCEWithLogitsLoss`
   - Only applies to ש

4. **Stress Classifier** (binary)
   - Predicts stress marks (◌֫)
   - Uses `BCEWithLogitsLoss`

### Why This Design?

Vowels are **mutually exclusive** (a letter can't have both patah and tsere), so we use multi-class classification. However, other marks like dagesh, sin, and stress are **independent** and can combine with vowels, so they use binary classification.

## Data Encoding

The model processes mixed text (Hebrew, English, numbers, punctuation) but only predicts nikud for Hebrew letters. Here's how the data is encoded:

### Example Input
```python
plain_text = "האיש hello 123 רצה"  # Everything preserved!
```

### Label Structure

Each token gets 4 labels:

```python
labels = [
    {'vowel': 1, 'dagesh': 0, 'sin': 0, 'stress': 0},  # ה
    {'vowel': 3, 'dagesh': 0, 'sin': 0, 'stress': 0},  # א
    {'vowel': 0, 'dagesh': 0, 'sin': 0, 'stress': 0},  # י
    {'vowel': 0, 'dagesh': 0, 'sin': 0, 'stress': 0},  # ש
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # space (ignored)
    
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # h (ignored)
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # e (ignored)
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # l (ignored)
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # l (ignored)
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # o (ignored)
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # space (ignored)
    
    {'vowel': 1, 'dagesh': 0, 'sin': 0, 'stress': 0},  # ר
    {'vowel': 1, 'dagesh': 0, 'sin': 0, 'stress': 0},  # צ
    {'vowel': 0, 'dagesh': 0, 'sin': 0, 'stress': 0},  # ה
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # space (ignored)
    
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # 1 (ignored)
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # 2 (ignored)
    {'vowel': -100, 'dagesh': -100, 'sin': -100, 'stress': -100},  # 3 (ignored)
]
```

### Key Points

- **Hebrew letters** get actual label values (0-5 for vowels, 0/1 for binary marks)
- **Non-Hebrew characters** (English, numbers, spaces, punctuation) get `-100` for all labels
- The `-100` value tells the loss function to **ignore** these positions
- This allows the model to handle mixed text naturally while only learning Hebrew nikud

## Training Process

### Loss Functions

1. **Vowel Loss**: `CrossEntropyLoss` with label smoothing (0.1)
   - Label smoothing prevents overconfidence and helps with class imbalance
   - Automatically ignores positions with label `-100`

2. **Binary Losses**: `BCEWithLogitsLoss` for dagesh, sin, and stress
   - Applied with masking to ignore `-100` positions
   - Each loss is computed independently

3. **Combined Loss**: Simple sum of all 4 losses
   ```
   total_loss = vowel_loss + dagesh_loss + sin_loss + stress_loss
   ```

### Training Configuration

- **Optimizer**: AdamW with learning rate ~1e-4
- **Gradient Clipping**: Max norm of 1.0
- **Batch Size**: Typically 8 (adjustable)
- **Evaluation Metrics**: WER (Word Error Rate) and CER (Character Error Rate)

## Inference Process

### Step-by-Step

1. **Tokenization**: Input text is tokenized using the character-level tokenizer
   ```python
   encoding = tokenizer(text, return_tensors='pt')
   ```

2. **Model Forward Pass**: Text goes through BERT + 4 classification heads
   - BERT produces contextualized embeddings for each token
   - Each head produces logits for its task

3. **Prediction Decoding**:
   - **Vowels**: `argmax` over 6 classes → class 0-5
   - **Binary marks**: `sigmoid + threshold (0.5)` → 0 or 1

4. **Text Reconstruction**: Predictions are converted back to nikud characters and combined with the original text
   - Hebrew letters get their predicted nikud marks
   - Non-Hebrew characters remain unchanged

### Example

```python
from inference import NikudPredictor

predictor = NikudPredictor('checkpoints/best_model.pt')
result = predictor.predict("האיש רצה")
print(result)  # הָאִישׁ רָצָה
```

## Key Design Decisions

### 1. Character-Level Tokenization
Hebrew diacritics are added to individual characters, so character-level tokenization is essential. DictaBERT's character-level approach is perfect for this task.

### 2. Hybrid Classification
Vowels are mutually exclusive (one per letter max), while dagesh, sin, and stress can combine. Using separate loss functions for each type allows the model to learn these different behaviors.

### 3. Label Smoothing
Set to 0.1 for the vowel classifier to prevent overconfidence and improve generalization, especially for rare vowel patterns.

### 4. Sentinel Value (-100)
Using `-100` for non-Hebrew tokens allows the model to:
- Handle mixed Hebrew/English/number text naturally
- Only compute loss on Hebrew letters
- Preserve all non-Hebrew content during inference

### 5. NFD Normalization
Hebrew text uses NFD (Canonical Decomposition) normalization to ensure consistent representation of base letters and combining marks.

## File References

- **`src/model.py`** - Model architecture and forward pass
- **`src/trainer.py`** - Custom training loop with WER/CER metrics
- **`src/inference.py`** - Inference wrapper and text reconstruction
- **`src/constants.py`** - Unicode definitions for nikud marks
- **`src/prepare_data.py`** - Data preprocessing pipeline

