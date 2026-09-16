"""Normalize CUAD contract text without losing the annotation offsets.

CUAD marks each answer span with character positions into the raw contract
text, so any edit to that text moves the positions.  This module cleans the
text and reports, for every surviving character, where it used to live.  A
span can then be moved onto the cleaned text instead of being thrown away.

Every rule here only deletes characters or replaces a run with a shorter run
of characters that the run already started with.  Nothing invents new text.
That restriction is what makes the position map exact.  The one exception is
``SPACE_VARIANTS``, which swaps single characters one-for-one and so cannot
move anything either.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter


# Typographic spaces that a PDF extractor emits where the contract just had
# a space.  Each maps to one ASCII space, so the text length never changes.
SPACE_VARIANTS = str.maketrans({
    "\u00a0": " ",  # no-break space
    "\u2002": " ",  # en space
    "\u2003": " ",  # em space
    "\u2009": " ",  # thin space
    "\u200a": " ",  # hair space
    "\u202f": " ",  # narrow no-break space
})

# (name, pattern, replacement).  Order matters: line-level rules run before
# the rules that collapse whitespace, so a removed page marker cannot leave a
# blank line behind.
RULES: list[tuple[str, re.Pattern[str], str]] = [
    # PDF hyphenation leaves invisible soft hyphens inside words.
    ("soft_hyphen", re.compile("\u00ad"), ""),
    # Invisible and meaningless, but tokenizers still count it.
    ("zero_width_space", re.compile("\u200b"), ""),
    # Page furniture such as "Page 4 of 17" on its own line.  The trailing
    # newline goes too, so a page break falling mid-sentence does not leave a
    # blank line that would later read as a paragraph boundary.
    ("page_marker", re.compile(r"(?im)^[ \t]*page\s+\d+\s+of\s+\d+[ \t]*\n?"), ""),
    # Table-of-contents dot leaders.  One dot survives so that the entry and
    # its page number stay apart: "DEFINITIONS.......1" must not become
    # "DEFINITIONS1".
    ("dot_leaders", re.compile(r"\.{4,}"), "."),
    ("trailing_whitespace", re.compile(r"[ \t]+$", re.MULTILINE), ""),
    # Column artifacts from the PDF extraction: "THIS  AGREEMENT  is  made".
    ("collapsed_spaces", re.compile(r"[ \t]{2,}"), " "),
    # Keep paragraph breaks, drop the rest.
    ("collapsed_blank_lines", re.compile(r"\n{3,}"), "\n\n"),
]


MAX_PASSES = 5


def _clean_once(text: str) -> tuple[str, list[int], Counter[str]]:
    """Apply every rule once, all of them matching against ``text``."""
    changes: Counter[str] = Counter()

    # One-for-one swaps first, so the collapse rules below see plain spaces.
    translated = text.translate(SPACE_VARIANTS)
    if translated != text:
        changes["space_variants"] = sum(a != b for a, b in zip(text, translated))
        text = translated

    keep = bytearray(b"\x01") * len(text)
    for name, pattern, replacement in RULES:
        for match in pattern.finditer(text):
            start, end = match.span()
            # Keep the first len(replacement) characters of the match and drop
            # the rest.  Each rule's replacement is a prefix of what it
            # matches, so the kept characters are already the right ones.
            for index in range(start + len(replacement), end):
                if keep[index]:
                    keep[index] = 0
                    changes[name] += 1

    cleaned = []
    origins = []
    for index, keeping in enumerate(keep):
        if keeping:
            cleaned.append(text[index])
            origins.append(index)
    return "".join(cleaned), origins, changes


def clean_with_map(text: str) -> tuple[str, list[int], Counter[str]]:
    """Clean ``text`` and record where each surviving character came from.

    Returns the cleaned text, a list ``origins`` where ``origins[i]`` is the
    index in ``text`` of the ``i``-th cleaned character, and a count of how
    many characters each rule changed.  Every count is a deletion except
    ``space_variants``, which is a one-for-one substitution.

    Rules within a pass all match the same text, so one rule cannot see what
    another just removed: deleting a page marker can push two blank lines
    together and leave a run that the blank-line rule already looked past.
    Passes therefore repeat until the text stops changing, composing the
    position maps as they go.
    """
    cleaned, origins, changes = _clean_once(text)

    for _ in range(MAX_PASSES - 1):
        if not changes:
            break
        cleaned, again, changes_again = _clean_once(cleaned)
        if not changes_again:
            break
        # Compose the maps: a character now at i sat at again[i] in the
        # previous pass, which in turn came from origins[again[i]].
        origins = [origins[index] for index in again]
        changes += changes_again

    return cleaned, origins, changes


def remap_span(start: int, end: int, origins: list[int]) -> tuple[int, int]:
    """Move a ``[start, end)`` span from the raw text onto the cleaned text.

    Characters deleted at the edges of the span fall away, and characters
    deleted inside it close up, so a span survives cleaning that happens to
    run through the middle of it.
    """
    return bisect.bisect_left(origins, start), bisect.bisect_left(origins, end)
