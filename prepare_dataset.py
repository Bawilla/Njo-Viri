"""
Balanda Dataset Preparation
============================
Extracts, cleans, deduplicates, normalises Unicode, validates, and splits
all translation pairs from the Njo-Viri JSONL corpus into:
  train.jsonl  (85 %)
  val.jsonl    (10 %)
  test.jsonl   ( 5 %)

Each output record: {"instruction": str, "input": str, "output": str}
"""

import glob
import json
import os
import random
import unicodedata

# ── ASCII substitutes → proper Unicode codepoints ─────────────────────────────
ASCII_SUBS = {
    # eng
    "ng'": "\u014b",   # ŋ  (fallback: digraph)
    # vowels with diaeresis
    "ii":  "ï",        # common ASCII stand-in in some files
    "uu":  "ü",
    "oo":  "ö",
}

UNICODE_CHARS = {"ŋ", "ï", "ü", "ö"}   # targets we want to preserve / fix


def normalise_unicode(text: str) -> str:
    """NFC-normalise and repair known ASCII substitutes."""
    # NFC first
    text = unicodedata.normalize("NFC", text)
    # Repair explicit ASCII substitution patterns only in Balanda text
    # (conservative: only replace ng' → ŋ where it appears as a word-final or
    # inter-vowel digraph – we do a simple global replace here; adjust if needed)
    text = text.replace("ng\u2019", "\u014b").replace("ng'", "\u014b")
    return text


def parse_file(path: str) -> list[dict]:
    """
    Parse one JSONL file and return a list of raw dicts.
    Handles two formats:
      A) Each line is a standalone JSON object (simple pairs)
      B) File contains one or more multi-line JSON objects / arrays
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read().strip()

    records = []
    lines = content.splitlines()
    buf: list[str] = []

    for line in lines:
        buf.append(line)
        try:
            obj = json.loads("\n".join(buf))
            records.append(obj)
            buf = []
        except json.JSONDecodeError:
            pass  # keep accumulating

    # Leftover line-by-line (simple format)
    for line in buf:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    return records


def extract_pairs(records: list) -> list[dict]:
    """
    Flatten parsed records into instruction/input/output dicts.
    Handles:
      - {"instruction", "input", "output"} directly
      - list of {"balanda", "english"} pairs
      - dict with grammar_rule wrapper containing the list as second record
    """
    pairs = []
    for rec in records:
        if isinstance(rec, dict):
            # Direct instruction/input/output record
            if {"instruction", "input", "output"} <= rec.keys():
                pairs.append({
                    "instruction": rec["instruction"],
                    "input":       rec["input"],
                    "output":      rec["output"],
                })
            # Balanda/English pair (sometimes appears as standalone dict)
            elif {"balanda", "english"} <= rec.keys():
                pairs.append({
                    "instruction": "Translate from Balanda to English",
                    "input":       rec["balanda"],
                    "output":      rec["english"],
                })
        elif isinstance(rec, list):
            for item in rec:
                if isinstance(item, dict) and {"balanda", "english"} <= item.keys():
                    pairs.append({
                        "instruction": "Translate from Balanda to English",
                        "input":       item["balanda"],
                        "output":      item["english"],
                    })
    return pairs


def clean(pair: dict) -> dict | None:
    """
    Validate and normalise a single pair.
    Returns None if the record should be dropped.
    """
    for field in ("instruction", "input", "output"):
        if field not in pair:
            return None
        if not isinstance(pair[field], str) or not pair[field].strip():
            return None

    return {
        "instruction": normalise_unicode(pair["instruction"].strip()),
        "input":       normalise_unicode(pair["input"].strip()),
        "output":      normalise_unicode(pair["output"].strip()),
    }


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove exact duplicates keyed on (input, output)."""
    seen: set[tuple] = set()
    unique = []
    for p in pairs:
        key = (p["input"].lower(), p["output"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def split(pairs: list[dict], train_pct=0.85, val_pct=0.10, seed=42):
    """Deterministic shuffle then split."""
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_pct)
    n_val   = int(n * val_pct)
    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]
    return train, val, test


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    jsonl_files = glob.glob(os.path.join(repo_root, "**", "*.jsonl"), recursive=True)

    # Exclude the output files themselves
    output_names = {"train.jsonl", "val.jsonl", "test.jsonl"}
    jsonl_files = [
        f for f in jsonl_files
        if os.path.basename(f) not in output_names
        and "prepare_dataset" not in f
    ]

    print(f"Found {len(jsonl_files)} source JSONL files.")

    raw_pairs: list[dict] = []
    for path in sorted(jsonl_files):
        records = parse_file(path)
        pairs = extract_pairs(records)
        raw_pairs.extend(pairs)
        print(f"  {os.path.relpath(path, repo_root):<65}  {len(pairs):>4} pairs")

    print(f"\nRaw pairs extracted : {len(raw_pairs)}")

    # Clean
    cleaned = [c for p in raw_pairs if (c := clean(p)) is not None]
    print(f"After cleaning      : {len(cleaned)}")

    # Deduplicate
    unique = deduplicate(cleaned)
    print(f"After deduplication : {len(unique)}")

    # Split
    train, val, test = split(unique)
    print(f"\nSplit  →  train={len(train)}  val={len(val)}  test={len(test)}")

    # Write
    for name, subset in [("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test)]:
        out_path = os.path.join(repo_root, name)
        write_jsonl(out_path, subset)
        print(f"  Saved {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
