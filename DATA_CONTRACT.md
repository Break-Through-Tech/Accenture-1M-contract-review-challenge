# Clean CUAD data contract

Run the preparation script from the project folder:

```powershell
python scripts/prepare_cuad.py
```

It reads the official train/test JSON files and writes four files under
`data/processed/`.

| File | One row represents | Main fields |
| --- | --- | --- |
| `contracts.jsonl` | one contract | `contract_id`, `text`, `split`, `n_chars`, `n_lines`, `n_chars_removed`, `is_duplicate_of` |
| `annotations.jsonl` | one contract and clause category | `contract_id`, `category`, `annotations`, `has_annotation`, `qa_ids`, `split` |
| `categories.json` | one of the 41 clause categories | `name`, `description`, `answer_format`, `group` |
| `preparation_summary.json` | the run's checks | cleaning counts, category totals, invalid spans, duplicates |

`categories.json` is `category_descriptions.csv` with the `Category: `,
`Description: ` and similar prefixes stripped from every cell, which issue #8
noted needed doing. Cells are normalized before the prefix is stripped,
because one of them is `Answer Format:` followed by a no-break space. A cell
that is `-` or empty after that becomes `null`.

`annotations` is a list because one category can have more than one relevant
piece of text. Each annotation has `text`, `start`, `end`, `start_raw`,
`end_raw`, and `span_is_valid`. An empty list means the category was not
annotated in that contract. About 68% of rows are empty; those negatives are
what make precision and recall meaningful, so they are kept rather than
dropped.

## Getting the four fields in one table

The issue asks for `contract_id`, `text`, `category` and the annotation
fields together. Use the loader rather than a saved file:

```python
from scripts.load_dataset import load_clauses   # from the project folder

clauses = load_clauses(split="train")
clauses.columns
# contract_id, text, category, annotations, has_annotation, split
```

The join is built in memory on purpose. Saving it would repeat each contract
once per category and turn 26 MB of text into roughly 1.1 GB, over the
project's stated size budget. Storing the text once in `contracts.jsonl` and
joining on `contract_id` keeps it to 38 MB.

## Cleaning

The script normalizes the contract text and then moves every annotation
offset onto the cleaned text, so `text[start:end]` still returns the clause.
This is what the project overview asks for: normalize formatting, whitespace,
headers and page breaks, while preserving the spans that make a prediction
explainable.

| Rule | Action | Characters changed |
| --- | --- | --- |
| `collapsed_spaces` | runs of 2+ spaces or tabs become one space | 935,040 |
| `collapsed_blank_lines` | 3+ newlines become a paragraph break | 47,393 |
| `dot_leaders` | table-of-contents `....` runs become one `.`, so `DEFINITIONS.......1` reads `DEFINITIONS.1` rather than `DEFINITIONS1` | 14,728 |
| `page_marker` | `Page 4 of 17` lines, with the line break | 7,748 |
| `space_variants` | em/en/thin/hair/no-break spaces become an ASCII space | 693 |
| `soft_hyphen` | invisible PDF hyphenation characters | 121 |
| `zero_width_space` | invisible U+200B characters | 63 |
| `trailing_whitespace` | spaces at end of line | 15 |

That removes 3.75% of characters. Legal wording is never touched: no
lowercasing, no punctuation stripping, no stopword removal, no Unicode
folding, because defined terms like `"Agreement"` depend on their casing and
characters like `§`, `®` and `é` are part of the contract. The exact counts
for a run are in `preparation_summary.json` under `cleaning.changes_by_rule`.

Rules are applied in repeated passes until the text stops changing. A single
pass is not enough, because removing a page marker can push two blank lines
together after the blank-line rule has already run.

### How the offsets survive

Every rule only deletes characters or shortens a run to a prefix of itself,
and `space_variants` swaps single characters one-for-one. Nothing new is
ever inserted and nothing ever moves. That lets `clean_text.clean_with_map`
return, for each surviving character, the position it held in the raw text,
and a span is then remapped with a binary search over that list. Characters
deleted at the edge of a span fall away; characters deleted inside it close
up, so a clause split by a page break stays one span.

The invisible characters in `clean_text.py` are written as `\u00ad`-style
escapes rather than literally, and a test enforces that, so a reviewer can
actually see what each rule matches.

Output files use `\n` line endings on every OS, so a run on Windows and a
run on macOS produce byte-identical files.

`start_raw` and `end_raw` keep the original CUAD offsets on every annotation,
so any span can still be traced back to the source JSON.

## Checks

`prepare_cuad.py` re-validates all 13,823 spans against the cleaned text and
records the result in `preparation_summary.json` as `invalid_span_count`,
which is currently 0. Run the tests with:

```powershell
python -m pytest
```

`tests/test_clean_text.py` covers the offset remapping on small synthetic
contracts, plus one integration check over the real prepared files.

## Known data notes

* **Split sizes are the official ones**: 408 train and 102 test contracts, as
  released by The Atticus Project. Do not re-split them.
* **One duplicate contract.** `ADUROBIOTECH,INC_06_02_2020-EX-10.7-CONSULTING
  AGREEMENT` has text identical to the `(1)` copy, both in train. It is
  flagged with `is_duplicate_of` rather than deleted, so the official counts
  stay comparable to the CUAD paper. Filter it out during modelling if it
  matters.
* **A validation set does not exist yet.** It has to be carved out of the 408
  train contracts, leaving the official test set untouched.
