#!/usr/bin/env python3
"""
prepare_dataset.py — Balanda (Njo Viri) fine-tuning dataset builder.

Auto-discovers all *_v1.jsonl files in the repo, optionally augments records
from grammar-rule headers, deduplicates, applies a stratified 90/5/5 split,
and writes train.jsonl / val.jsonl / test.jsonl.

Usage
-----
    python prepare_dataset.py                           # augmentation on (default)
    python prepare_dataset.py --no-augment-from-rules  # disable augmentation
    python prepare_dataset.py --seed 0 --out-dir /tmp  # custom seed / output dir
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

# ── Repo root (directory containing this script) ──────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent

# ── Keyword → human grammar category (first match wins) ──────────────────────
_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("conjunction",   "conjunction"),
    ("connector",     "connector"),
    ("modal",         "modal"),
    ("pronoun",       "pronoun"),
    ("possessive",    "pronoun"),
    ("adjective",     "adjective"),
    ("locative",      "locative"),
    ("demonstrative", "demonstrative"),
    ("interrogative", "interrogative"),
    ("greeting",      "greeting"),
    ("preposition",   "preposition"),
    ("alphabet",      "letter"),
    ("orthography",   "letter"),
    ("phonology",     "letter"),
    ("lexicon",       "word"),
    ("infinitive",    "verb"),
    ("verb",          "verb"),
    ("number",        "number"),
    ("noun",          "noun"),
    ("continuous",    "tense"),
    ("perfect",       "tense"),
    ("going_to",      "tense"),
    ("tense",         "tense"),
    ("simple",        "tense"),
    ("affirmative",   "tense"),
    ("negation",      "tense"),
    ("question",      "tense"),
    ("past",          "tense"),
    ("future",        "tense"),
    ("present",       "tense"),
]


def grammar_name_to_category(grammar_name: str) -> str:
    name = grammar_name.lower()
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in name:
            return category
    return "grammar item"


# ── Streaming JSON parser ─────────────────────────────────────────────────────

def stream_objects(path: str):
    """
    Yield all JSON objects from *path* using raw_decode.
    Handles pretty-printed headers, compact record lines, JSON arrays, and
    interleaved non-JSON text (e.g. section labels in older files).
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    dec = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        rest = content[idx:].lstrip(" \t\r\n")
        if not rest:
            break
        skipped = len(content[idx:]) - len(rest)
        try:
            obj, sz = dec.raw_decode(rest)
            idx += skipped + sz
            if isinstance(obj, list):
                yield from (x for x in obj if isinstance(x, dict))
            elif isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            nl = content.find("\n", idx)
            if nl == -1:
                break
            idx = nl + 1


# ── Classification ────────────────────────────────────────────────────────────

def is_header(obj: dict) -> bool:
    return obj.get("instruction_type") == "grammar_rule"


def is_record(obj: dict) -> bool:
    return "instruction" in obj and "input" in obj and "output" in obj


def get_instruction_type(instruction: str) -> str:
    if instruction == "Translate from Balanda to English":
        return "translate_b2e"
    if instruction == "Translate from English to Balanda":
        return "translate_e2b"
    if instruction.startswith("Identify"):
        return "identify"
    if instruction.startswith("Explain"):
        return "explain"
    return "other"


# ── Rule augmentation ─────────────────────────────────────────────────────────

def augment_from_header(header: dict, source_file: str) -> list[dict]:
    """
    For each key in header['rules'], produce one 'Explain' training record.
    Output format: "<key> \u2014 <function>. <notes> Structure: <structure>."
    """
    grammar_name = header.get("grammar_name", "")
    category = grammar_name_to_category(grammar_name)
    rules = header.get("rules", {})
    augmented: list[dict] = []
    for key, val in rules.items():
        if not isinstance(val, dict):
            continue
        function = val.get("function", "").strip()
        structure = val.get("structure", "").strip()
        notes = val.get("notes", "").strip()
        parts = [f"{key} \u2014 {function}."]
        if notes:
            parts.append(notes)
        if structure:
            parts.append(f"Structure: {structure}.")
        augmented.append({
            "instruction": f"Explain the Balanda {category} {key}",
            "input": key,
            "output": " ".join(parts),
            "_source": source_file,
            "_itype": "explain",
        })
    return augmented


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_sources(
    repo_root: Path,
    augment: bool,
) -> tuple[list[dict], dict[str, int]]:
    """
    Discover all *_v1.jsonl files, parse, classify, (optionally) augment.
    Returns (records_with_provenance, per_source_primary_counts).
    Each internal record has: instruction, input, output, _source, _itype.
    """
    pattern = str(repo_root / "**" / "*_v1.jsonl")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        sys.exit("No *_v1.jsonl files found — check REPO_ROOT.")

    all_records: list[dict] = []
    per_source: dict[str, int] = {}

    for path in paths:
        rel = str(Path(path).relative_to(repo_root))
        file_records: list[dict] = []
        header: dict | None = None

        for obj in stream_objects(path):
            if is_header(obj):
                if header is None:
                    header = obj
            elif is_record(obj):
                itype = get_instruction_type(obj["instruction"])
                file_records.append({
                    "instruction": obj["instruction"],
                    "input": obj["input"],
                    "output": obj["output"],
                    "_source": rel,
                    "_itype": itype,
                })

        per_source[rel] = len(file_records)
        all_records.extend(file_records)

        if augment and header is not None:
            all_records.extend(augment_from_header(header, rel))

    return all_records, per_source


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for r in records:
        key = (r["instruction"], r["input"], r["output"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique, len(records) - len(unique)


# ── Stratified 90/5/5 split ───────────────────────────────────────────────────

def stratified_split(
    records: list[dict],
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split by (source_file, instruction_type) stratum.
    Strata with < 10 items fall back to a random proportional split.
    """
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        strata[(rec["_source"], rec["_itype"])].append(rec)

    train_all: list[dict] = []
    val_all: list[dict] = []
    test_all: list[dict] = []

    rng = random.Random(seed)

    for _key, items in strata.items():
        rng.shuffle(items)
        n = len(items)

        if n < 2:
            train_all.extend(items)
            continue

        if n < 10:
            # Random proportional fallback (no sklearn)
            n_test = max(1, round(n * test_frac))
            n_val = max(1, round(n * val_frac))
            if n_test + n_val >= n:
                n_val = max(0, n - n_test - 1)
            test_all.extend(items[:n_test])
            val_all.extend(items[n_test : n_test + n_val])
            train_all.extend(items[n_test + n_val :])
        else:
            tv, test = train_test_split(items, test_size=test_frac, random_state=seed)
            val_ratio = val_frac / (train_frac + val_frac)
            train, val = train_test_split(tv, test_size=val_ratio, random_state=seed)
            train_all.extend(train)
            val_all.extend(val)
            test_all.extend(test)

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    return train_all, val_all, test_all


# ── Output ────────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            out = {
                "instruction": rec["instruction"],
                "input": rec["input"],
                "output": rec["output"],
            }
            fh.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_jsonl(path: str) -> int:
    count = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"Validation failed {path}:{lineno}: {exc}")
            if set(obj.keys()) != {"instruction", "input", "output"}:
                sys.exit(
                    f"Validation failed {path}:{lineno}: "
                    f"unexpected keys {set(obj.keys())}"
                )
            count += 1
    return count


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(
    per_source: dict[str, int],
    n_total_raw: int,
    n_dedup: int,
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> None:
    print("\n\u2550\u2550\u2550 Source file counts (primary records) \u2550\u2550\u2550")
    col = max(len(k) for k in per_source) + 2
    for src, cnt in sorted(per_source.items()):
        print(f"  {src:<{col}} {cnt:>5}")

    n_aug = n_total_raw - sum(per_source.values())
    print(f"\n  {'Augmented from rules':<{col}} {n_aug:>5}")
    print(f"  {'TOTAL (before dedup)':<{col}} {n_total_raw:>5}")
    print(f"  {'Duplicates removed':<{col}} {n_dedup:>5}")
    print(f"  {'TOTAL (after dedup)':<{col}} {n_total_raw - n_dedup:>5}")

    total = len(train) + len(val) + len(test)
    print(f"\n\u2550\u2550\u2550 Split sizes \u2550\u2550\u2550")
    print(f"  train  {len(train):>6}  ({100*len(train)/total:.1f}%)")
    print(f"  val    {len(val):>6}  ({100*len(val)/total:.1f}%)")
    print(f"  test   {len(test):>6}  ({100*len(test)/total:.1f}%)")

    def itype_dist(split: list[dict]) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for r in split:
            d[r["_itype"]] += 1
        return dict(d)

    all_itypes = sorted({r["_itype"] for r in train + val + test})
    print(f"\n\u2550\u2550\u2550 Instruction-type distribution per split \u2550\u2550\u2550")
    col2 = max(len(t) for t in all_itypes) + 2
    print(f"  {'type':<{col2}} {'train':>7} {'val':>6} {'test':>6}")
    td, vd, sd = itype_dist(train), itype_dist(val), itype_dist(test)
    for it in all_itypes:
        print(f"  {it:<{col2}} {td.get(it, 0):>7} {vd.get(it, 0):>6} {sd.get(it, 0):>6}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=".",
        help="Directory for output JSONL files (default: repo root)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--augment-from-rules",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Augment training data from grammar-rule headers (default: on)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo root    : {REPO_ROOT}")
    print(f"Out dir      : {out_dir}")
    print(f"Seed         : {args.seed}")
    print(f"Augmentation : {args.augment_from_rules}")

    records, per_source = load_all_sources(REPO_ROOT, augment=args.augment_from_rules)
    n_total_raw = len(records)

    records, n_dedup = deduplicate(records)

    train, val, test = stratified_split(records, seed=args.seed)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = str(out_dir / f"{split_name}.jsonl")
        write_jsonl(split_data, path)
        n = validate_jsonl(path)
        print(f"Wrote {path}  ({n} records validated)")

    print_summary(per_source, n_total_raw, n_dedup, train, val, test)


if __name__ == "__main__":
    main()
