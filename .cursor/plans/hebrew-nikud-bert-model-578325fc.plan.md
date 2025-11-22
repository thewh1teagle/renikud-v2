<!-- 578325fc-ad32-4029-a546-72c467241dec ff3219fa-6d6d-4c41-bd35-72d94b96c2a2 -->
# Hebrew Nikud BERT Model

## Overview

Fine-tune the dictabert-large-char model to predict Hebrew diacritization marks (nikud) for each character. The model will use 4 separate classification heads for vowel type, dagesh, sin, and stress marks.

## Implementation Steps

### 1. Add Dependencies

Update `pyproject.toml` to include:

- `torch`
- `transformers`
- `datasets` (for data handling)
- `accelerate` (for training optimization)

### 2. Create Dataset Preparation Module (`src/dataset.py`)

Create a module that:

- Takes nikud'd Hebrew text as input
- Strips nikud to create model input (plain Hebrew letters)
- Extracts nikud labels for each character:
- **Vowel labels**: 6 classes (0=None, 1=PATAH, 2=TSERE, 3=HIRIK, 4=HOLAM, 5=QUBUT)
- **Dagesh labels**: binary (0=no, 1=yes) - only for letters in `CAN_HAVE_DAGESH`
- **Sin labels**: binary (0=no, 1=yes) - only for 'ש'
- **Stress labels**: binary (0=no, 1=yes)
- Handle character alignment (map each input character to its labels)
- Use existing `normalize()` function to preprocess text

### 3. Create Model Architecture (`src/model.py`)

Build a custom model class that:

- Loads `dicta-il/dictabert-large-char` as base
- Adds 4 separate classification heads on top:
- Vowel head: Linear(hidden_size, 6)
- Dagesh head: Linear(hidden_size, 2)
- Sin head: Linear(hidden_size, 2)
- Stress head: Linear(hidden_size, 2)
- Implements forward pass with masking for invalid predictions (e.g., dagesh only on בכפו, sin only on ש)
- Returns separate losses for each head

### 4. Create Training Script (`src/train.py`)

Implement training logic:

- Initialize model and tokenizer from `dicta-il/dictabert-large-char`
- Create single-example dataset with: `"הַאִיש שֵלֹא רַצַה לִהיוֹת אַגַדַה"`
- Implement custom training loop or use HuggingFace Trainer
- Combined loss = weighted sum of 4 classification losses
- Overfit on single example to verify model can learn (training sanity check)
- Save model checkpoints

### 5. Create Inference Script (`src/inference.py`)

Build inference module that:

- Loads trained model
- Takes plain Hebrew text input
- Predicts nikud for each character
- Reconstructs nikud'd text by adding predicted marks
- Respects rules (no dagesh where not allowed, etc.)

### 6. Update Training Data

Prepare `data/train_nikud.txt` with the test sentence (already nikud'd)

## Key Files to Create/Modify

- `pyproject.toml` - add ML dependencies
- `src/dataset.py` - dataset preparation (NEW)
- `src/model.py` - model architecture (NEW)
- `src/train.py` - training script (NEW)
- `src/inference.py` - inference script (NEW)
- `data/train.txt` - training data (NEW)

## Success Criteria

The model should successfully overfit on the single training example, predicting all nikud marks correctly for: `"האיש שלא רצה להיות אגדה"`

## How It Actually Works (Simple Explanation)

### What is BERT?

**BERT** (Bidirectional Encoder Representations from Transformers) is an **encoder-only** architecture, which means:

- **Input**: Text goes in
- **Output**: Rich contextual representations for each character/word
- **Not a generator**: Unlike GPT (decoder), BERT doesn't generate text. It "understands" text by encoding it into vectors that capture meaning and context.

Think of BERT as a reading comprehension expert. You give it a sentence, and it deeply "understands" each character by looking at all the surrounding context (both left and right, hence "bidirectional").

### What is DictaBERT-large-char?

`dicta-il/dictabert-large-char` is:

1. **A BERT model** pre-trained specifically for Hebrew
2. **Character-level**: Unlike word-level tokenizers, it treats each Hebrew letter as a separate token
3. **Large**: Uses the BERT-Large architecture (24 layers, 1024 hidden size, 16 attention heads)
4. **Pre-trained**: Already learned Hebrew language patterns from massive amounts of text

The base model was trained with **Masked Language Modeling (MLM)**: randomly hide some characters and teach the model to predict them from context. This teaches it Hebrew grammar, word structure, and letter patterns.

### Our Fine-Tuning Approach

We take this pre-trained BERT encoder and add **4 classification heads** on top:

```
Input Text: "האיש שלא"
     ↓
[Character Tokenizer] → ['[CLS]', 'ה', 'א', 'י', 'ש', ' ', 'ש', 'ל', 'א', '[SEP]']
     ↓
[BERT Encoder - 24 layers] → Rich vector for each character (1024 dimensions)
     ↓
Split into 4 parallel heads:
     ├─→ [Vowel Classifier] → 6 classes (None/PATAH/TSERE/HIRIK/HOLAM/QUBUT)
     ├─→ [Dagesh Classifier] → 2 classes (No/Yes)
     ├─→ [Sin Classifier] → 2 classes (No/Yes)
     └─→ [Stress Classifier] → 2 classes (No/Yes)
```

### Training Process

**What we train:**

- The **4 classification heads** (randomly initialized, learn from scratch)
- **Fine-tune** the BERT encoder (already pre-trained, just adjust weights)

**How:**

1. **Input**: Plain Hebrew text without nikud (`"האיש שלא"`)
2. **Labels**: Ground truth nikud for each character (extracted from nikud'd text)
3. **Forward Pass**: 

   - BERT encodes each character into a rich vector
   - Each classification head predicts its label from that vector

4. **Loss Calculation**: Compare predictions to ground truth for all 4 tasks
5. **Backpropagation**: Update weights to minimize loss
6. **Repeat**: Until the model learns the patterns

**Key insight**: Because BERT is bidirectional, when predicting nikud for 'א', the model can see:

- The letter itself ('א')
- Previous letters ('ה')
- Following letters ('י', 'ש')
- Word boundaries (spaces)
- The entire sentence context

This context is crucial because Hebrew nikud often depends on grammar, word position, and surrounding words!

### Encoder vs Decoder

- **Encoder (what we use)**: Processes input text to understand it. Good for classification tasks like "what nikud should this letter have?"
- **Decoder (not used here)**: Generates new text token by token. Good for tasks like translation or text generation.
- **Encoder-Decoder (not used here)**: Combination used for seq2seq tasks like translation.

For nikud prediction, we only need to **classify** each character (encoder task), not **generate** new text (decoder task).

### Why This Works

1. **Pre-training**: BERT already learned Hebrew patterns from millions of texts
2. **Character-level**: Sees fine-grained structure of Hebrew words
3. **Bidirectional context**: Uses full sentence context to disambiguate nikud
4. **Multi-task learning**: Learning vowels, dagesh, sin, and stress together helps the model learn richer representations
5. **Rule enforcement**: We mask invalid predictions (e.g., dagesh only on בכפו) so the model never learns impossible patterns

### Training Results

On a single training example, the model achieved:

- **100% accuracy** on all 4 tasks
- Loss dropped from 2.78 → 0.003 in 100 epochs
- Successfully learned to map plain text → nikud'd text

This proves the architecture works! Next step: scale to larger datasets.

### To-dos

- [x] Add torch, transformers, datasets, accelerate to pyproject.toml
- [x] Create dataset.py with nikud extraction and label encoding
- [x] Create model.py with 4-head classification architecture
- [x] Create train.py with training loop for single example
- [x] Create inference.py for prediction and text reconstruction
- [x] Verify model can overfit on single training example