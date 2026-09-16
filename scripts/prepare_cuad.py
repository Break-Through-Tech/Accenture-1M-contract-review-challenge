"""Create clean, easy-to-use CUAD contract and annotation files.

The original CUAD files are SQuAD-style JSON.  This script separates them into
two smaller, linked JSONL files instead of repeating a long contract 41 times:

* contracts.jsonl: one row per contract (contract_id, text, split)
* annotations.jsonl: one row per contract/category (category, annotations)

Join the two files later with ``contract_id`` if a model needs both text and a
category label in the same table, or call ``scripts/load_dataset.py``.

The contract text is normalized on the way through (see ``clean_text.py``) and
every annotation offset is moved onto the cleaned text, so the spans keep
pointing at the clause they describe.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clean_text import clean_with_map, remap_span


QUESTION_CATEGORY = re.compile(r'related to "([^"]+)"')


def category_from_question(question: str) -> str:
    """Get the category name embedded in CUAD's standard question text."""
    match = QUESTION_CATEGORY.search(question)
    if not match:
        raise ValueError(f"Could not find a category in question: {question!r}")
    return match.group(1).strip()


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)["data"]


def load_categories(path: Path) -> list[dict[str, str]]:
    """Read category_descriptions.csv into clean rows.

    Every cell in the CSV repeats its column name as a prefix, for example
    ``"Description: The name of the contract"``, so the prefixes are stripped
    here once instead of by every notebook that reads the file.  The cells
    also carry the same stray no-break spaces as the contracts, so they go
    through the same normalizer.
    """
    columns = {
        "Category (incl. context and answer)": ("name", "Category:"),
        "Description": ("description", "Description:"),
        "Answer Format": ("answer_format", "Answer Format:"),
        "Group": ("group", "Group:"),
    }
    categories = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            clean = {}
            for column, (field, prefix) in columns.items():
                # Normalize first: one cell is "Answer Format:" followed by a
                # no-break space, which a prefix with an ASCII space misses.
                value = clean_with_map(row[column])[0]
                value = value.removeprefix(prefix).strip()
                # "-" and an empty cell are both the CSV's way of saying none.
                clean[field] = value if value and value != "-" else None
            categories.append(clean)
    return categories


def canonical_category_lookup(categories: list[dict[str, str]]) -> dict[str, str]:
    """Map case variations from questions to CUAD's official category names."""
    return {category["name"].casefold(): category["name"] for category in categories}


def prepare_split(
    source_path: Path,
    split: str,
    canonical_categories: dict[str, str],
    contracts_file: Any,
    annotations_file: Any,
    stats: Counter[str],
    category_counts: Counter[str],
    failures: list[dict[str, Any]],
    changes_by_rule: Counter[str],
    seen_texts: dict[str, str],
    duplicates: list[dict[str, str]],
) -> None:
    for contract in read_json(source_path):
        contract_id = contract["title"].strip()
        paragraphs = contract["paragraphs"]

        # CUAD normally has one paragraph per contract. Supporting more than one
        # keeps this script safe if a future version changes that layout.
        for paragraph_number, paragraph in enumerate(paragraphs):
            raw_text = paragraph["context"]
            text, origins, changes = clean_with_map(raw_text)
            changes_by_rule.update(changes)
            stats["characters_before_cleaning"] += len(raw_text)
            paragraph_id = (
                contract_id
                if len(paragraphs) == 1
                else f"{contract_id}__paragraph_{paragraph_number}"
            )

            # Two CUAD entries can hold the same contract. The official split
            # sizes are quoted in the project overview and are meant to match
            # the CUAD paper, so flag the repeat rather than dropping a row.
            duplicate_of = seen_texts.get(text)
            if duplicate_of is None:
                seen_texts[text] = paragraph_id
            else:
                duplicates.append(
                    {"contract_id": paragraph_id, "duplicate_of": duplicate_of}
                )

            contract_row = {
                "contract_id": paragraph_id,
                "text": text,
                "split": split,
                "n_chars": len(text),
                "n_lines": text.count("\n") + 1,
                "n_chars_removed": len(raw_text) - len(text),
                "is_duplicate_of": duplicate_of,
            }
            contracts_file.write(json.dumps(contract_row, ensure_ascii=False) + "\n")
            stats[f"{split}_contracts"] += 1
            stats["characters_after_cleaning"] += len(text)

            # CUAD can use several QA entries for the same category, one for
            # each answer span. Grouping produces one clean record per category.
            grouped: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"annotations": [], "qa_ids": []}
            )
            for qa in paragraph["qas"]:
                extracted_category = category_from_question(qa["question"])
                try:
                    category = canonical_categories[extracted_category.casefold()]
                except KeyError as error:
                    raise ValueError(
                        f"Question category is absent from category_descriptions.csv: "
                        f"{extracted_category!r}"
                    ) from error
                group = grouped[category]
                group["qa_ids"].append(qa["id"])

                for answer in qa["answers"]:
                    raw_answer = answer["text"]
                    raw_start = answer["answer_start"]
                    raw_end = raw_start + len(raw_answer)

                    # Move the span onto the cleaned text, then compare against
                    # the answer cleaned by the same rules. Comparing against
                    # the raw answer would fail wherever cleaning touched the
                    # clause itself, for example by collapsing double spaces.
                    start, end = remap_span(raw_start, raw_end, origins)
                    expected, _, _ = clean_with_map(raw_answer)
                    is_valid = text[start:end] == expected
                    if not is_valid:
                        failures.append(
                            {
                                "contract_id": paragraph_id,
                                "category": category,
                                "qa_id": qa["id"],
                                "start": start,
                                "expected": expected,
                                "found": text[start:end],
                            }
                        )
                    group["annotations"].append(
                        {
                            "text": expected,
                            "start": start,
                            "end": end,
                            "start_raw": raw_start,
                            "end_raw": raw_end,
                            "span_is_valid": is_valid,
                        }
                    )

            for category, group in grouped.items():
                annotations = group["annotations"]
                annotation_row = {
                    "contract_id": paragraph_id,
                    "category": category,
                    "annotations": annotations,
                    "has_annotation": bool(annotations),
                    "qa_ids": group["qa_ids"],
                    "split": split,
                }
                annotations_file.write(
                    json.dumps(annotation_row, ensure_ascii=False) + "\n"
                )
                stats[f"{split}_contract_category_rows"] += 1
                stats[f"{split}_answer_spans"] += len(annotations)
                category_counts[category] += len(annotations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw CUAD JSON into clean contract and annotation JSONL files."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/cuad"), help="Folder with CUAD JSON files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed"), help="Folder for cleaned files"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    changes_by_rule: Counter[str] = Counter()
    seen_texts: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []

    categories = load_categories(args.input_dir / "category_descriptions.csv")
    canonical_categories = canonical_category_lookup(categories)
    categories_path = args.output_dir / "categories.json"
    categories_path.write_text(
        json.dumps(categories, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # newline="\n" keeps the record separators identical on every OS; without
    # it Windows writes "\r\n" and the files stop being byte-for-byte
    # reproducible between teammates.
    with (
        (args.output_dir / "contracts.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as contracts_file,
        (args.output_dir / "annotations.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as annotations_file,
    ):
        for source_name, split in [
            ("train_separate_questions.json", "train"),
            ("test.json", "test"),
        ]:
            prepare_split(
                args.input_dir / source_name,
                split,
                canonical_categories,
                contracts_file,
                annotations_file,
                stats,
                category_counts,
                failures,
                changes_by_rule,
                seen_texts,
                duplicates,
            )

    stats["total_contracts"] = stats["train_contracts"] + stats["test_contracts"]
    stats["total_answer_spans"] = (
        stats["train_answer_spans"] + stats["test_answer_spans"]
    )
    characters_before = stats.pop("characters_before_cleaning")
    characters_after = stats.pop("characters_after_cleaning")
    characters_removed = characters_before - characters_after

    summary = {
        "source_files": [
            "train_separate_questions.json",
            "test.json",
            "category_descriptions.csv",
        ],
        "cleaning_rule": (
            "Contract text is normalized and every annotation offset is remapped "
            "onto the cleaned text, so text[start:end] still returns the clause."
        ),
        "cleaning": {
            "characters_before": characters_before,
            "characters_after": characters_after,
            "characters_removed": characters_removed,
            "percent_removed": round(100 * characters_removed / characters_before, 2),
            # Every entry is a deletion count except space_variants, which
            # counts one-for-one swaps of typographic spaces to ASCII.
            "changes_by_rule": dict(sorted(changes_by_rule.items())),
        },
        "stats": dict(stats),
        "categories": len(category_counts),
        "answer_spans_by_category": dict(sorted(category_counts.items())),
        "invalid_span_count": len(failures),
        "invalid_spans": failures,
        "duplicate_contract_count": len(duplicates),
        "duplicate_contracts": duplicates,
    }
    summary_path = args.output_dir / "preparation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Wrote {args.output_dir / 'contracts.jsonl'}")
    print(f"Wrote {args.output_dir / 'annotations.jsonl'}")
    print(f"Wrote {categories_path}")
    print(f"Contracts: {stats['train_contracts']} train, {stats['test_contracts']} test")
    print(f"Categories: {len(category_counts)}")
    print(
        f"Cleaning: removed {characters_removed:,} of {characters_before:,} characters "
        f"({summary['cleaning']['percent_removed']}%)"
    )
    print(f"Answer spans: {stats['total_answer_spans']:,}")
    print(f"Invalid answer spans: {len(failures)}")
    print(f"Duplicate contracts flagged: {len(duplicates)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
