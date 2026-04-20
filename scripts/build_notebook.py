"""Generate balanda_finetune.ipynb from cell definitions."""
import json, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "balanda_finetune.ipynb"

def cell_id(): return str(uuid.uuid4())[:8]

def md(src):
    return {"cell_type": "markdown", "id": cell_id(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "id": cell_id(), "metadata": {},
            "source": src, "outputs": [], "execution_count": None}


# ── Cell 1: Title + overview ──────────────────────────────────────────────────
C1 = md("""\
# Balanda (Njo Viri) — QLoRA Fine-tuning with Qwen2.5-1.5B-Instruct

Fine-tunes **[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)** \
on the Balanda (Njo Viri) language dataset using QLoRA (4-bit nf4 + PEFT) and **TRL SFTTrainer**.  \n\
Source data and dataset pipeline: **[Bawilla/Njo-Viri](https://github.com/Bawilla/Njo-Viri)**.  \n\
Target runtime: **Colab free T4** (fp16, batch=2, grad-accum=8, seq-len=1024).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bawilla/Njo-Viri/blob/main/balanda_finetune.ipynb)

---
**RUN_TRAINING / RUN_EVAL / PUSH_TO_HUB** flags in the Config cell are all `False` by default.  \n\
Flip them manually before running the relevant cells.""")


# ── Cell 2: Environment setup ─────────────────────────────────────────────────
C2 = code("""\
import sys, os

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U",
         "transformers", "datasets", "peft", "trl", "bitsandbytes",
         "accelerate", "sacrebleu", "sentencepiece", "huggingface_hub",
         "scikit-learn"],
        check=True,
    )
    from google.colab import drive
    drive.mount("/content/drive")
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
else:
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("Set HF_TOKEN in your environment before running locally.")

import huggingface_hub
huggingface_hub.login(token=os.environ["HF_TOKEN"])

import torch, transformers, peft, trl, bitsandbytes as bnb
print(f"torch          {torch.__version__}")
print(f"transformers   {transformers.__version__}")
print(f"peft           {peft.__version__}")
print(f"trl            {trl.__version__}")
print(f"bitsandbytes   {bnb.__version__}")""")


# ── Cell 3: Config ────────────────────────────────────────────────────────────
C3 = code("""\
CFG = {
    # Base model
    "BASE_MODEL":           "Qwen/Qwen2.5-1.5B-Instruct",

    # LoRA
    "LORA_R":               16,
    "LORA_ALPHA":           32,
    "LORA_DROPOUT":         0.05,
    "LORA_TARGET_MODULES":  "all-linear",

    # Training
    "MAX_SEQ_LEN":          1024,
    "EPOCHS":               3,
    "BATCH_SIZE":           2,
    "GRAD_ACCUM":           8,
    "LR":                   2e-4,
    "LR_SCHEDULER":         "cosine",
    "WARMUP_RATIO":         0.03,
    "SEED":                 42,

    # HuggingFace Hub
    "HF_MODEL_REPO":        "Bawilla/qwen2_5_1_5b_balanda_qlora",
    "HF_PRIVATE":           False,

    # Paths
    "OUTPUT_DIR":           "./outputs/qwen2_5_1_5b_balanda_qlora",
    "DRIVE_BACKUP_DIR":     "/content/drive/MyDrive/balanda_qlora",

    # Guards — flip to True manually before running each cell
    "RUN_TRAINING":         False,
    "RUN_EVAL":             False,
    "PUSH_TO_HUB":          False,
}

RUN_TRAINING = CFG["RUN_TRAINING"]
RUN_EVAL     = CFG["RUN_EVAL"]
PUSH_TO_HUB  = CFG["PUSH_TO_HUB"]

print("CFG loaded.")
for k, v in CFG.items():
    print(f"  {k:<28} {v}")""")


# ── Cell 4: Get repo + load splits ───────────────────────────────────────────
C4 = code("""\
import os
from pathlib import Path
from datasets import load_dataset

if IS_COLAB and not Path("Njo-Viri").exists():
    os.system("git clone https://github.com/Bawilla/Njo-Viri.git")
    os.chdir("Njo-Viri")
elif IS_COLAB and Path("Njo-Viri").exists():
    os.chdir("Njo-Viri")

dataset = load_dataset(
    "json",
    data_files={
        "train":      "train.jsonl",
        "validation": "val.jsonl",
        "test":       "test.jsonl",
    },
)

print(dataset)
for split in ("train", "validation", "test"):
    print(f"\\n── {split} ({len(dataset[split])}) ──")
    print(dataset[split][0])""")


# ── Cell 5: Format function ───────────────────────────────────────────────────
C5 = code("""\
SYSTEM_PROMPT = (
    "You are a Balanda (Njo Viri) language expert. "
    "Follow the user's instruction precisely."
)

def format_example(ex: dict) -> str:
    user_content = ex["instruction"]
    if ex.get("input"):
        user_content += "\\n\\n" + ex["input"]
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": ex["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

try:
    print(format_example(dataset["train"][0]))
except NameError:
    print("(Re-run this cell after loading the tokenizer in the next cell.)")""")


# ── Cell 6: Tokenizer + 4-bit model ──────────────────────────────────────────
C6 = code("""\
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(CFG["BASE_MODEL"], trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    CFG["BASE_MODEL"],
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

print("Base model loaded.")
print(f"  dtype  : {next(model.parameters()).dtype}")
print(f"  device : {next(model.parameters()).device}")""")


# ── Cell 7: LoRA ─────────────────────────────────────────────────────────────
C7 = code("""\
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=CFG["LORA_R"],
    lora_alpha=CFG["LORA_ALPHA"],
    lora_dropout=CFG["LORA_DROPOUT"],
    target_modules=CFG["LORA_TARGET_MODULES"],
    task_type="CAUSAL_LM",
    bias="none",
)

model = get_peft_model(model, lora_cfg)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable : {trainable:,}  ({100 * trainable / total:.2f}% of {total:,})")""")


# ── Cell 8: Trainer ───────────────────────────────────────────────────────────
C8 = code("""\
from trl import SFTConfig, SFTTrainer

sft_cfg = SFTConfig(
    output_dir=CFG["OUTPUT_DIR"],
    num_train_epochs=CFG["EPOCHS"],
    per_device_train_batch_size=CFG["BATCH_SIZE"],
    per_device_eval_batch_size=CFG["BATCH_SIZE"],
    gradient_accumulation_steps=CFG["GRAD_ACCUM"],
    learning_rate=CFG["LR"],
    lr_scheduler_type=CFG["LR_SCHEDULER"],
    warmup_ratio=CFG["WARMUP_RATIO"],
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    seed=CFG["SEED"],
    push_to_hub=CFG["PUSH_TO_HUB"],
    hub_model_id=CFG["HF_MODEL_REPO"],
    hub_private_repo=CFG["HF_PRIVATE"],
    hub_strategy="every_save",
    max_seq_length=CFG["MAX_SEQ_LEN"],
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    formatting_func=format_example,
)

print("SFTTrainer ready.")
print(f"  Train samples : {len(dataset['train'])}")
print(f"  Eval samples  : {len(dataset['validation'])}")""")


# ── Cell 9: Train ─────────────────────────────────────────────────────────────
C9 = code("""\
if RUN_TRAINING:
    import shutil

    trainer.train()
    trainer.save_model(CFG["OUTPUT_DIR"])
    tokenizer.save_pretrained(CFG["OUTPUT_DIR"])
    print(f"Adapter saved to {CFG['OUTPUT_DIR']}")

    if IS_COLAB:
        import os, shutil
        os.makedirs(CFG["DRIVE_BACKUP_DIR"], exist_ok=True)
        shutil.copytree(CFG["OUTPUT_DIR"], CFG["DRIVE_BACKUP_DIR"], dirs_exist_ok=True)
        print(f"Backed up to {CFG['DRIVE_BACKUP_DIR']}")
else:
    print("Training skipped — set RUN_TRAINING = True in CFG to run.")""")


# ── Cell 10: Eval ─────────────────────────────────────────────────────────────
C10 = code("""\
if RUN_EVAL:
    import random, torch, sacrebleu

    model.eval()

    def generate_batch(prompts, max_new_tokens=128):
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=CFG["MAX_SEQ_LEN"]).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new = [ids[enc["input_ids"].shape[1]:] for ids in out]
        return tokenizer.batch_decode(new, skip_special_tokens=True)

    BATCH = 8
    results_b2e, results_e2b = [], []

    for i in range(0, len(dataset["test"]), BATCH):
        batch = dataset["test"][i : i + BATCH]
        for j, instr in enumerate(batch["instruction"]):
            if instr not in ("Translate from Balanda to English",
                             "Translate from English to Balanda"):
                continue
            ex = {k: batch[k][j] for k in batch}
            prompt_ex = {**ex, "output": ""}
            hyp = generate_batch([format_example(prompt_ex)])[0].strip()
            ref = ex["output"].strip()
            (results_b2e if "Balanda to English" in instr else results_e2b).append((ref, hyp))

    def score(pairs, label):
        if not pairs:
            print(f"{label}: no samples"); return
        refs = [[r] for r, _ in pairs]
        hyps = [h for _, h in pairs]
        chrf = sacrebleu.corpus_chrf(hyps, refs, word_order=2).score
        bleu = sacrebleu.corpus_bleu(hyps, refs).score
        print(f"\\n{label}  (n={len(pairs)})")
        print(f"  chrF++  {chrf:.2f}   BLEU  {bleu:.2f}")
        print("\\nRandom samples (ref | hyp):")
        for ref, hyp in random.sample(pairs, min(10, len(pairs))):
            print(f"  REF: {ref}")
            print(f"  HYP: {hyp}"); print()

    score(results_b2e, "Balanda → English")
    score(results_e2b, "English → Balanda")
else:
    print("Eval skipped — set RUN_EVAL = True in CFG to run.")""")


# ── Cell 11: Push to Hub ──────────────────────────────────────────────────────
C11_src = '''\
if PUSH_TO_HUB:
    import huggingface_hub, os

    trainer.push_to_hub(commit_message="QLoRA adapter for Balanda (Njo Viri)")
    tokenizer.push_to_hub(CFG["HF_MODEL_REPO"], private=CFG["HF_PRIVATE"])

    model_card = f"""---
base_model: {CFG['BASE_MODEL']}
library_name: peft
license: apache-2.0
language:
  - en
  - und
tags:
  - lora
  - qlora
  - balanda
  - njo-viri
  - translation
  - instruction-tuning
datasets:
  - Bawilla/Njo-Viri
---

# Qwen2.5-1.5B-Instruct — Balanda (Njo Viri) QLoRA Adapter

QLoRA fine-tune of [{CFG['BASE_MODEL']}](https://huggingface.co/{CFG['BASE_MODEL']})
on the [Bawilla/Njo-Viri](https://github.com/Bawilla/Njo-Viri) dataset covering Balanda
grammar rules, translation pairs, and identification exercises.

## Dataset
70 *_v1.jsonl source files (tenses, modalities, pronouns, nouns, prepositions,
conjunctions, numbers, phonology, bilingual lexicon ~1 300 entries).
After deduplication: train {len(dataset["train"])} | val {len(dataset["validation"])} | test {len(dataset["test"])}.

## Training hyperparameters
| Parameter | Value |
|-----------|-------|
| Base model | `{CFG['BASE_MODEL']}` |
| LoRA r / alpha | {CFG['LORA_R']} / {CFG['LORA_ALPHA']} |
| LoRA dropout | {CFG['LORA_DROPOUT']} |
| Target modules | {CFG['LORA_TARGET_MODULES']} |
| Max seq length | {CFG['MAX_SEQ_LEN']} |
| Epochs | {CFG['EPOCHS']} |
| Batch / grad-accum | {CFG['BATCH_SIZE']} / {CFG['GRAD_ACCUM']} |
| Learning rate | {CFG['LR']} ({CFG['LR_SCHEDULER']}) |
| Warmup ratio | {CFG['WARMUP_RATIO']} |
| Precision | fp16 (Colab T4) |
| Seed | {CFG['SEED']} |

## Reload example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "{CFG['BASE_MODEL']}", load_in_4bit=True, device_map="auto",
)
model = PeftModel.from_pretrained(base, "{CFG['HF_MODEL_REPO']}")
tok = AutoTokenizer.from_pretrained("{CFG['HF_MODEL_REPO']}")

msgs = [
    {{"role": "system",    "content": "You are a Balanda (Njo Viri) language expert. Follow the user\'s instruction precisely."}},
    {{"role": "user",      "content": "Translate from Balanda to English\\n\\nNi timande kee ni ja nja gba."}},
]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# Expected: She works but she does not eat.
```
"""

    os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)
    card_path = os.path.join(CFG["OUTPUT_DIR"], "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    huggingface_hub.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=CFG["HF_MODEL_REPO"],
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"Pushed to https://huggingface.co/{CFG['HF_MODEL_REPO']}")
else:
    print("Hub push skipped — set PUSH_TO_HUB = True in CFG to push.")'''
C11 = code(C11_src)


# ── Cell 12a: Inference markdown ──────────────────────────────────────────────
C12a = md("""\
## Reload adapter for inference

After training and pushing, reload the adapter from the Hub:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    load_in_4bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "Bawilla/qwen2_5_1_5b_balanda_qlora")
tok = AutoTokenizer.from_pretrained("Bawilla/qwen2_5_1_5b_balanda_qlora")
```""")


# ── Cell 12b: Inference code ──────────────────────────────────────────────────
C12b = code("""\
import torch

DEMO = [
    ("Translate from Balanda to English",  "Ni timande kee ni ja nja gba."),
    ("Translate from Balanda to English",  "Ne leyai agbi maraya andiso ni merego adheke."),
    ("Translate from English to Balanda",  "She goes to the market with the children."),
    ("Identify the Balanda conjunction",   "wamina"),
]

def run_inference(instruction: str, input_text: str, max_new_tokens: int = 64) -> str:
    user = instruction + ("\\n\\n" + input_text if input_text else "")
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

model.eval()
for instr, inp in DEMO:
    print(f"Instruction : {instr}")
    print(f"Input       : {inp}")
    print(f"Output      : {run_inference(instr, inp)}")
    print()""")


# ── Assemble ──────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12a, C12b],
}

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"Wrote {OUT}  ({len(nb['cells'])} cells)")
