#!/usr/bin/env python3
"""
Normalize all *.jsonl files in the repo to the unified fine-tuning schema.

Target schema
-------------
Record 1 — header (pretty-printed, indent=2):
  { "instruction_type": "grammar_rule",
    "grammar_name": "<snake_case>",
    "description": "<paragraph>",
    "rules": { "<item>": { "function": "...", "structure": "...", "notes": "..." }, ... } }

Records 2..N — training examples (compact, one per line):
  { "instruction": "Translate from Balanda to English"|"Translate from English to Balanda"|"Identify the Balanda <cat>",
    "input": "...",
    "output": "..." }
"""

import json
import os
import re
import sys
import glob
from pathlib import Path

SCRIPT_PATH = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))  # scripts/../

CANONICAL_INSTRUCTIONS = {
    "Translate from Balanda to English",
    "Translate from English to Balanda",
}

IDENTIFY_RE = re.compile(
    r"^Identify\s+(?:the\s+)?(?:Balanda\s+)?(.+)$", re.IGNORECASE
)


def humanize(key: str) -> str:
    return key.replace("_", " ").replace("-", " ").strip()


def normalize_rules_value(key: str, value) -> dict:
    """Convert any rules value into {function, structure, notes}."""
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys <= {"function", "structure", "notes"}:
            return {
                "function": value.get("function", ""),
                "structure": value.get("structure", ""),
                "notes": value.get("notes", ""),
            }
        func = value.get("function", humanize(key))
        struct = (
            value.get("structure")
            or value.get("statement")
            or value.get("rule")
            or ""
        )
        used = {"function", "structure", "statement", "rule"}
        notes_parts = []
        for k, v in value.items():
            if k in used:
                continue
            if isinstance(v, str) and v:
                notes_parts.append(v)
            elif isinstance(v, list):
                flat = "; ".join(str(x) for x in v if not isinstance(x, (dict, list)))
                if flat:
                    notes_parts.append(flat)
        notes = " | ".join(notes_parts)
        if not struct:
            struct = json.dumps(value, ensure_ascii=False)
        return {"function": func, "structure": struct, "notes": notes}
    elif isinstance(value, str):
        return {"function": humanize(key), "structure": value, "notes": ""}
    elif isinstance(value, list):
        items = [str(x) for x in value if not isinstance(x, (dict, list))]
        return {"function": humanize(key), "structure": "; ".join(items), "notes": ""}
    else:
        return {"function": humanize(key), "structure": str(value), "notes": ""}


def normalize_header(obj: dict, filename: str) -> dict:
    grammar_name = obj.get("grammar_name") or filename_to_grammar_name(filename)
    description = obj.get("description", "")
    raw_rules = obj.get("rules", {})
    if isinstance(raw_rules, dict):
        rules = {k: normalize_rules_value(k, v) for k, v in raw_rules.items()}
    else:
        rules = {}
    return {
        "instruction_type": "grammar_rule",
        "grammar_name": grammar_name,
        "description": description,
        "rules": rules,
    }


def filename_to_grammar_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_v\d+$", "", stem)
    stem = re.sub(r"[\s\-]+", "_", stem)
    return stem.lower()


def normalize_instruction(instruction: str) -> str:
    if instruction in CANONICAL_INSTRUCTIONS:
        return instruction
    m = IDENTIFY_RE.match(instruction)
    if m:
        category = m.group(1).strip()
        if not category.lower().startswith("balanda"):
            return f"Identify the Balanda {category}"
        return f"Identify the {category}"
    return instruction  # non-canonical — preserve verbatim


def normalize_record(obj: dict) -> dict | None:
    """Convert any training-record format to {instruction, input, output}."""
    if "instruction" in obj and "input" in obj and "output" in obj:
        return {
            "instruction": normalize_instruction(obj["instruction"]),
            "input": obj["input"],
            "output": obj["output"],
        }
    # instruction + output only (no input) — keep with empty input
    if "instruction" in obj and "output" in obj and "input" not in obj:
        return {
            "instruction": normalize_instruction(obj["instruction"]),
            "input": "",
            "output": obj["output"],
        }
    # balanda/english format
    if "balanda" in obj and "english" in obj:
        return {
            "instruction": "Translate from Balanda to English",
            "input": obj["balanda"],
            "output": obj["english"],
        }
    # Lexicon translation_pair
    if obj.get("type") == "translation_pair" and "source" in obj and "target" in obj:
        return {
            "instruction": "Translate from Balanda to English",
            "input": obj["source"],
            "output": obj["target"],
        }
    return None


def infer_header(filename: str, records: list[dict]) -> dict:
    grammar_name = filename_to_grammar_name(filename)
    categories = set()
    for r in records:
        instr = r.get("instruction", "")
        m = IDENTIFY_RE.match(instr)
        if m:
            categories.add(m.group(1).strip())
    desc = (
        f"This file contains Balanda language training examples for {grammar_name.replace('_', ' ')}. "
        + ("Items covered: " + ", ".join(sorted(categories)) + "." if categories else "")
    ).strip()
    rules = {}
    if categories:
        for cat in sorted(categories):
            k = re.sub(r"\s+", "_", cat.lower())
            rules[k] = {"function": cat, "structure": "", "notes": "Inferred from training records."}
    return {
        "instruction_type": "grammar_rule",
        "grammar_name": grammar_name,
        "description": desc,
        "rules": rules,
    }


def infer_header_from_metadata(meta: dict, filename: str) -> dict:
    grammar_name = meta.get("dataset", filename_to_grammar_name(filename))
    grammar_name = re.sub(r"\s+", "_", grammar_name).lower()
    description = meta.get("description", f"Balanda lexicon extracted from {filename}.")
    return {
        "instruction_type": "grammar_rule",
        "grammar_name": grammar_name,
        "description": description,
        "rules": {},
    }


def parse_file(path: str):
    """Return (header_dict_or_None, list_of_raw_record_dicts, is_metadata_based)."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    dec = json.JSONDecoder()
    idx = 0
    raw_items = []  # mix of dicts and lists

    while idx < len(content):
        rest = content[idx:].lstrip(" \t\r\n")
        if not rest:
            break
        skipped = len(content[idx:]) - len(rest)
        try:
            obj, sz = dec.raw_decode(rest)
            idx += skipped + sz
            raw_items.append(obj)
        except json.JSONDecodeError:
            nl = content.find("\n", idx)
            if nl == -1:
                break
            idx = nl + 1

    header = None
    records = []
    metadata = None

    for item in raw_items:
        if isinstance(item, list):
            # Flatten array of training records
            for elem in item:
                if isinstance(elem, dict):
                    records.append(elem)
        elif isinstance(item, dict):
            if "instruction_type" in item or "grammar_name" in item:
                if header is None:
                    header = item
            elif item.get("type") == "metadata":
                metadata = item
            else:
                records.append(item)

    if header is None and metadata is not None:
        header = metadata  # will be detected as metadata type below

    return header, records, metadata


def process_file(path: str) -> tuple[str, int, int]:
    filename = os.path.basename(path)
    header_raw, record_objs, metadata = parse_file(path)

    inferred = False

    if header_raw is None:
        header = infer_header(filename, record_objs)
        inferred = True
    elif metadata is not None and header_raw is metadata:
        header = infer_header_from_metadata(metadata, filename)
        inferred = True
    else:
        header = normalize_header(header_raw, filename)

    norm_records = []
    for obj in record_objs:
        r = normalize_record(obj)
        if r is not None:
            norm_records.append(r)

    records_before = len(record_objs)
    records_after = len(norm_records)

    lines = [json.dumps(header, ensure_ascii=False, indent=2)]
    for r in norm_records:
        lines.append(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
    output = "\n".join(lines) + "\n"

    try:
        with open(path, encoding="utf-8", newline="") as f:
            existing = f.read()
        if existing == output:
            return "skipped", records_before, records_after
    except Exception:
        pass

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(output)

    return ("header-inferred" if inferred else "converted", records_before, records_after)


def main():
    results = []
    inferred_files = []

    pattern = os.path.join(REPO_ROOT, "**", "*.jsonl")
    all_paths = sorted(glob.glob(pattern, recursive=True))

    for path in all_paths:
        norm_path = path.replace("\\", "/")
        if "/.git/" in norm_path or norm_path.endswith("/.git"):
            continue
        if os.path.abspath(path) == SCRIPT_PATH:
            continue

        rel = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
        try:
            status, before, after = process_file(path)
        except Exception as e:
            status = f"ERROR: {e}"
            before = after = 0

        results.append((rel, status, before, after))
        if status == "header-inferred":
            inferred_files.append(rel)

    col_w = max(len(r[0]) for r in results) + 2
    print(f"\n{'Path':<{col_w}} {'Status':<18} {'Before':>8} {'After':>7}")
    print("-" * (col_w + 37))
    counts: dict[str, int] = {}
    for rel, status, before, after in results:
        short = status if len(status) <= 18 else status[:15] + "..."
        print(f"{rel:<{col_w}} {short:<18} {before:>8} {after:>7}")
        key = "error" if status.startswith("ERROR") else status
        counts[key] = counts.get(key, 0) + 1

    n_conv = counts.get("converted", 0)
    n_inf = counts.get("header-inferred", 0)
    n_skip = counts.get("skipped", 0)

    print(f"\nSummary: {n_conv} converted | {n_inf} header-inferred | {n_skip} skipped")
    if inferred_files:
        print(f"\nFiles with inferred headers (review needed):")
        for f in inferred_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
