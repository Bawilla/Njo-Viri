# Njo-Viri — Balanda Language Dataset & Fine-tuning

A parallel corpus for **Balanda (Njo Viri)**, a Bantu language, with tools to
fine-tune a neural machine-translation model (English ↔ Balanda) on Kaggle.

---

## Dataset statistics

| Split | Records |
|-------|---------|
| train | 3 669   |
| val   |   431   |
| test  |   217   |
| **total** | **4 317** |

Records are deduplicated, Unicode-normalised (ŋ ï ü ö), and validated.
Each record: `{"instruction": "...", "input": "<Balanda>", "output": "<English>"}`.

---

## Repository layout

```
Njo-Viri/
├── prepare_dataset.py      # Step 1 – clean & split the corpus
├── balanda_finetune.ipynb  # Step 2 – Kaggle fine-tuning notebook
├── train.jsonl             # 85 % split (generated)
├── val.jsonl               # 10 % split (generated)
├── test.jsonl              #  5 % split (generated)
├── <topic folders>/        # Raw JSONL source files
└── README.md
```

---

## Quick start

### Step 1 — Regenerate train/val/test splits (optional)

Requires Python 3.10+ with no extra packages.

```bash
git clone https://github.com/Bawilla/Njo-Viri.git
cd Njo-Viri
python prepare_dataset.py
```

This will print extraction counts per file and write `train.jsonl`, `val.jsonl`,
`test.jsonl` into the repo root.

---

### Step 2 — Fine-tune on Kaggle

#### 2a. Create a Kaggle Dataset

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Name it exactly `balanda-dataset`
3. Upload `train.jsonl`, `val.jsonl`, `test.jsonl`
4. Click **Create**

#### 2b. Add your HuggingFace token as a Kaggle Secret

1. Open any Kaggle notebook → **Add-ons → Secrets**
2. Add a secret named `HF_TOKEN` with your HuggingFace write token
   (get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

#### 2c. Create and run the notebook

1. Upload `balanda_finetune.ipynb` to Kaggle
2. Attach the `balanda-dataset` dataset (Add data → Your datasets)
3. Enable **GPU T4 x2** accelerator
4. In **Cell 2 (Config)**, set `hf_repo_id` to `YOUR_HF_USERNAME/nllb-balanda-lora`
5. **Run All**

The notebook will:
- Install all dependencies
- Tokenise the dataset for `facebook/nllb-200-distilled-600M`
- Apply LoRA (r=16, target: q_proj + v_proj)
- Train for up to 5 epochs with early stopping (patience=2)
- Evaluate with **ChrF++** on val and test sets
- Push the adapter weights to HuggingFace Hub

Expected training time: ~2–3 hours on a T4 GPU.

---

## Inference after training

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch

base      = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
model     = PeftModel.from_pretrained(base, "YOUR_HF_USERNAME/nllb-balanda-lora")
tokenizer = AutoTokenizer.from_pretrained("YOUR_HF_USERNAME/nllb-balanda-lora")

def translate(text):
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")
    out    = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["zul_Latn"],
        max_new_tokens=128,
        num_beams=4,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(translate("I will eat fish"))   # → Ne leja se
```

---

## Model choice

| Model | Size | Notes |
|-------|------|-------|
| `facebook/nllb-200-distilled-600M` | 600M | **Default.** Best multilingual coverage; Zulu seed for Bantu morphology. |
| `Helsinki-NLP/opus-mt-en-mul` | ~300M | Lighter; change `lora_target_modules` to `("q", "v")` for Marian architecture. |

---

## Unicode notes

Balanda uses the following non-ASCII characters:

| Character | Unicode | Notes |
|-----------|---------|-------|
| ŋ | U+014B | Voiced velar nasal (eng) |
| ï | U+00EF | i with diaeresis |
| ü | U+00FC | u with diaeresis |
| ö | U+00F6 | o with diaeresis |

`prepare_dataset.py` NFC-normalises all text and converts the common ASCII
substitute `ng'` → `ŋ`.

---

## Citation

If you use this dataset, please cite:

```
@misc{njo-viri-2025,
  title  = {Njo-Viri: Balanda Language Parallel Corpus},
  author = {Bawilla},
  year   = {2025},
  url    = {https://github.com/Bawilla/Njo-Viri}
}
```
