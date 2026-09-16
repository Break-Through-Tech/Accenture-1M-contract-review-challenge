"""Checks on the pieces of prepare_cuad.py that do not need the CUAD files."""

from __future__ import annotations

from pathlib import Path

from prepare_cuad import canonical_category_lookup, load_categories

HEADER = "Category (incl. context and answer),Description,Answer Format,Group\n"


def write_csv(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "category_descriptions.csv"
    path.write_text(HEADER + rows, encoding="utf-8", newline="")
    return path


def test_prefixes_are_stripped_from_every_cell(tmp_path):
    path = write_csv(
        tmp_path,
        "Category: Document Name,Description: The name of the contract,"
        "Answer Format: Contract Name,Group: -\n",
    )
    assert load_categories(path) == [
        {
            "name": "Document Name",
            "description": "The name of the contract",
            "answer_format": "Contract Name",
            "group": None,
        }
    ]


def test_prefix_followed_by_no_break_space_becomes_none(tmp_path):
    # The real CSV has exactly this cell for Affiliate License-Licensee.
    path = write_csv(
        tmp_path,
        "Category: Affiliate License-Licensee,Description: Something,"
        "Answer Format:\u00a0,Group: 3\n",
    )
    [category] = load_categories(path)
    assert category["answer_format"] is None
    assert category["group"] == "3"


def test_no_break_spaces_inside_a_description_are_normalized(tmp_path):
    path = write_csv(
        tmp_path,
        "Category: Price Restrictions,Description: on the\u00a0 ability of a party,"
        "Answer Format: Yes/No,Group: 5\n",
    )
    [category] = load_categories(path)
    assert category["description"] == "on the ability of a party"


def test_lookup_is_case_insensitive(tmp_path):
    path = write_csv(
        tmp_path,
        "Category: Notice Period To Terminate Renewal,Description: x,"
        "Answer Format: y,Group: 1\n",
    )
    lookup = canonical_category_lookup(load_categories(path))
    assert lookup["notice period to terminate renewal"] == (
        "Notice Period To Terminate Renewal"
    )
