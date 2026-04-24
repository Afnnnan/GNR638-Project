# GNR638 Project: Deep Learning MCQ Answering from Images

Automated system for answering deep learning multiple-choice questions from PNG images using **Qwen3-VL-8B-Instruct**, a state-of-the-art Vision-Language Model with chain-of-thought reasoning.

## How It Works

1. Reads PNG images of deep learning MCQs (with LaTeX math, code snippets, diagrams)
2. Feeds each image to Qwen3-VL-8B with **thinking mode** enabled for step-by-step reasoning
3. Parses the model's answer (1–4) with a retry mechanism and hallucination guard
4. Outputs a `submission.csv` in the required format

## Testing on Kaggle (Step-by-Step)

### Step 1: Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Set **Accelerator** to **GPU** (T4 or P100 — the model fits on any 16GB+ GPU)
3. Enable **Internet** (needed for the first cell to install dependencies and download model)

### Step 2: Install Dependencies (Cell 1)

```python
# Run this cell FIRST with internet ON
!pip install -q git+https://github.com/huggingface/transformers accelerate qwen-vl-utils Pillow pandas

# Download the model weights (takes ~5-10 min on Kaggle)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
model_id = "Qwen/Qwen3-VL-8B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
print("✅ Model downloaded and loaded successfully!")
del model, processor  # free memory, will reload in the script
```

### Step 3: Upload Your Test Data

Upload your test dataset as a **Kaggle Dataset**. The structure should be:
```
your-dataset/
├── images/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
├── test.csv
└── sample_submission.csv
```

Or simply upload the folder via **Add Data** → **Upload** in the notebook sidebar.

### Step 4: Clone This Repo and Run (Cell 2)

```python
# Clone the repo
!git clone https://github.com/Afnnnan/GNR638-Project.git
%cd GNR638-Project

# Run inference — adjust the data_dir path to match your uploaded dataset
# If your dataset is added as a Kaggle dataset:
!python solve_mcq.py --data_dir /kaggle/input/your-dataset-name --output /kaggle/working/submission.csv

# If you uploaded files directly:
# !python solve_mcq.py --data_dir /kaggle/input/ --output /kaggle/working/submission.csv
```

### Step 5: Verify Output (Cell 3)

```python
import pandas as pd
df = pd.read_csv("/kaggle/working/submission.csv")
print(df)
print(f"\nTotal questions: {len(df)}")
print(f"Answered: {(df['option'] != 5).sum()}")
print(f"Abstained: {(df['option'] == 5).sum()}")
print(f"Answer distribution:\n{df['option'].value_counts().sort_index()}")
```

### Step 6: Turn Off Internet & Re-run (Final Submission)

For the actual competition submission:
1. **Turn OFF internet** in notebook settings
2. The model weights are already cached from Step 2
3. Re-run Cell 2 and Cell 3 — everything runs from local cache
4. Download `submission.csv` from the output

### Important Notes for Kaggle

| Setting | Value |
|---|---|
| **GPU** | Any GPU (T4/P100/L40s) — model uses ~16GB VRAM in float16 |
| **Internet** | ON for setup (Cell 1), can be OFF for inference |
| **Runtime** | ~30-60s per question with thinking mode |
| **Max questions** | 50 questions in < 1 hour |

### Alternative: Run Everything in One Cell

```python
# === CELL 1: Setup + Run (Internet ON) ===
!pip install -q git+https://github.com/huggingface/transformers accelerate qwen-vl-utils Pillow pandas
!git clone https://github.com/Afnnnan/GNR638-Project.git 2>/dev/null; true
%cd GNR638-Project

# Change this path to your test data location
DATA_DIR = "/kaggle/input/your-dataset-name"

!python solve_mcq.py --data_dir {DATA_DIR} --output /kaggle/working/submission.csv

import pandas as pd
print(pd.read_csv("/kaggle/working/submission.csv"))
```

---

## Local Usage

### Setup

```bash
# Create environment
conda create -n gnr638 python=3.11 -y
conda activate gnr638

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Basic run (downloads model on first run)
python solve_mcq.py --data_dir ./sample_test_project_2

# With custom output path
python solve_mcq.py --data_dir ./sample_test_project_2 --output my_submission.csv

# Disable thinking mode (faster, less accurate)
python solve_mcq.py --data_dir ./sample_test_project_2 --no_think

# Use local model weights
python solve_mcq.py --data_dir ./sample_test_project_2 --model_path /path/to/local/weights
```

### Run Tests (no model needed)

```bash
python test_pipeline.py
# Expected: 58/58 tests passed
```

## Output Format

```csv
image_name,option
image_1,1
image_2,3
```

- `1–4`: Answer corresponding to options A–D
- `5`: Abstain (no answer, 0 points)

## Scoring

| Outcome | Points |
|---|---|
| Correct | +1 |
| Incorrect | −0.25 |
| Abstain (5) | 0 |
| Hallucinated (not 1-5) | −1 |

The pipeline has strict validation to ensure outputs are always in {1,2,3,4,5}, preventing hallucination penalties.

## Model Details

- **Model**: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (Sep 2025)
- **Architecture**: Vision-Language Model with DeepStack multi-level ViT fusion
- **Thinking Mode**: Chain-of-thought reasoning via `<think>` tags
- **VRAM**: ~16GB in float16
- **OCR**: 32-language support, robust on LaTeX math and code

## References

```bibtex
@misc{qwen3technicalreport,
    title={Qwen3 Technical Report},
    author={Qwen Team},
    year={2025},
    eprint={2505.09388},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.09388},
}
```
