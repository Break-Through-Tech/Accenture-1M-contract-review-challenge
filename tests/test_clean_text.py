"""Checks that cleaning never moves an annotation onto the wrong text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clean_text import clean_with_map, remap_span


def roundtrip(text: str, answer: str) -> str:
    """Clean ``text`` and return whatever the span for ``answer`` now covers."""
    start = text.index(answer)
    cleaned, origins, _ = clean_with_map(text)
    new_start, new_end = remap_span(start, start + len(answer), origins)
    return cleaned[new_start:new_end]


def test_text_without_artifacts_is_untouched():
    text = "THIS AGREEMENT is governed by the laws of Delaware.\n\nSigned."
    cleaned, origins, removed = clean_with_map(text)
    assert cleaned == text
    assert origins == list(range(len(text)))
    assert removed == {}


def test_span_after_a_page_marker_still_lands_on_its_clause():
    text = "Recitals.\nPage 2 of 9\nGoverning Law: New York law applies."
    assert roundtrip(text, "Governing Law: New York law applies.") == (
        "Governing Law: New York law applies."
    )


def test_span_containing_collapsed_spaces_is_cleaned_consistently():
    text = 'THIS  DISTRIBUTOR  AGREEMENT (the  "Agreement")'
    assert roundtrip(text, 'THIS  DISTRIBUTOR  AGREEMENT') == (
        "THIS DISTRIBUTOR AGREEMENT"
    )


def test_span_with_deletion_running_through_its_middle():
    # A page marker falling between two halves of one clause.
    text = "The term shall\nPage 3 of 8\ncontinue for five years."
    assert roundtrip(text, "The term shall\nPage 3 of 8\ncontinue for five years.") == (
        "The term shall\ncontinue for five years."
    )


def test_span_at_the_very_start():
    text = "DISTRIBUTOR AGREEMENT\n\n\n\nRecitals"
    assert roundtrip(text, "DISTRIBUTOR AGREEMENT") == "DISTRIBUTOR AGREEMENT"


def test_span_at_the_very_end():
    text = "Recitals....................\n\nGoverned by Illinois law."
    assert roundtrip(text, "Governed by Illinois law.") == "Governed by Illinois law."


def test_dot_leaders_keep_the_entry_and_page_number_apart():
    cleaned, _, changes = clean_with_map("DEFINITIONS.......1\nTERM.....12")
    assert cleaned == "DEFINITIONS.1\nTERM.12"
    assert changes["dot_leaders"] == 6 + 4


def test_ordinary_ellipsis_is_left_alone():
    cleaned, _, changes = clean_with_map("and so on... etc.")
    assert cleaned == "and so on... etc."
    assert "dot_leaders" not in changes


def test_soft_hyphens_are_removed_from_inside_a_word():
    text = "The Agree\u00adment is perpetual."
    cleaned, _, removed = clean_with_map(text)
    assert cleaned == "The Agreement is perpetual."
    assert removed["soft_hyphen"] == 1


def test_typographic_spaces_become_ascii_and_collapse():
    # em space, en space, hair space, narrow no-break space, no-break space
    text = "Term\u2003\u2002of\u200a\u202fthe\u00a0Agreement"
    cleaned, origins, changes = clean_with_map(text)
    assert cleaned == "Term of the Agreement"
    assert changes["space_variants"] == 5
    assert changes["collapsed_spaces"] == 2
    assert len(origins) == len(cleaned)


def test_span_over_typographic_spaces_still_matches():
    text = "Effective\u2003Date:\u200a\u200aJanuary 1, 2020"
    assert roundtrip(text, "Effective\u2003Date:\u200a\u200aJanuary 1, 2020") == (
        "Effective Date: January 1, 2020"
    )


def test_zero_width_space_is_deleted():
    cleaned, _, changes = clean_with_map("Con\u200bfidential")
    assert cleaned == "Confidential"
    assert changes["zero_width_space"] == 1


def test_source_file_contains_no_literal_invisible_characters():
    # The rules are written with \\u escapes on purpose: a literal soft hyphen
    # or zero-width space in the source is impossible to see in a review.
    source = (Path(__file__).resolve().parents[1] / "scripts" / "clean_text.py").read_text(
        encoding="utf-8"
    )
    assert all(ord(character) < 128 for character in source)


def test_blank_lines_collapse_to_a_paragraph_break():
    cleaned, _, _ = clean_with_map("Section 1.\n\n\n\n\nSection 2.")
    assert cleaned == "Section 1.\n\nSection 2."


def test_removing_a_page_marker_does_not_leave_a_blank_run_behind():
    # Deleting the marker pushes the blank lines together, which only the
    # next pass can see.
    text = "End of section.\n\nPage 7 of 12\n\nNext section."
    cleaned, _, _ = clean_with_map(text)
    assert cleaned == "End of section.\n\nNext section."


def test_spans_survive_a_multi_pass_clean():
    text = "End of section.\n\nPage 7 of 12\n\nNext  section  here."
    assert roundtrip(text, "Next  section  here.") == "Next section here."


def test_cleaning_reaches_a_fixed_point():
    text = "A.\n\nPage 1 of 3\n\nB....................\n\nPage 2 of 3\n\nC  D  "
    cleaned, _, _ = clean_with_map(text)
    twice, _, _ = clean_with_map(cleaned)
    assert cleaned == twice


def test_origins_are_strictly_increasing_and_in_range():
    text = "A  B\nPage 1 of 2\n\n\n\nC....................D  \nE"
    cleaned, origins, _ = clean_with_map(text)
    assert len(origins) == len(cleaned)
    assert all(origins[i] < origins[i + 1] for i in range(len(origins) - 1))
    assert all(text[origin] == character for origin, character in zip(origins, cleaned))


def test_empty_text():
    assert clean_with_map("") == ("", [], {})


PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.mark.slow
def test_every_span_in_the_prepared_dataset_is_valid():
    """The prepared files must agree with themselves: text[start:end] == span."""
    summary_path = PROCESSED / "preparation_summary.json"
    if not summary_path.exists():
        pytest.skip("run scripts/prepare_cuad.py first")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["invalid_span_count"] == 0

    contracts = {}
    with (PROCESSED / "contracts.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            contracts[row["contract_id"]] = row["text"]

    checked = 0
    with (PROCESSED / "annotations.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            text = contracts[row["contract_id"]]
            for span in row["annotations"]:
                assert text[span["start"] : span["end"]] == span["text"]
                checked += 1
    assert checked == summary["stats"]["total_answer_spans"]
