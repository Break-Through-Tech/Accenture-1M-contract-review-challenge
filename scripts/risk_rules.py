"""Rule-based clause risk scoring, Risk Rules v1 (issue #31).

Turns the risk-factor table in ``RISK_DESIGN.md`` into transparent Python rules.
Given the text of a detected clause, a rule reads a few facts out of it (who holds a
right, how many days of notice, whether a cap is mutual, ...) and maps the facts to
Low / Medium / High, with a one-sentence rationale and the rule id that fired.

    from risk_rules import score_clause, score_register

    result = score_clause("Termination for Convenience", span_text,
                          reviewer_aliases=("Licensee", "Acme"))
    result.level, result.rationale

    register = score_register({"Cap on Liability": [span1, span2], "Exclusivity": None})

Decisions worth knowing about (all of them are also written into each result):

* Risk is judged from the reviewing party's side. A clause does not say which party
  that is, so pass ``reviewer_aliases`` (party names or roles such as ``"Licensee"``).
  Without them the direction of a one-sided term is unknown and, by default, is read
  against the reviewer (``assume_adverse_when_direction_unknown``) so that High-risk
  clauses are not missed. The result lists that assumption.
* Low needs positive evidence. A detected clause the rules cannot read scores Medium
  under a rule id ending in ``.default``, never Low.
* A category that is not detected scores Unknown, not Low (the classifier misses some
  clauses). Unknown has score 0 so a contract-level roll-up can treat it explicitly.
* When one category has several spans, the highest level among the spans the rules
  could actually read wins; unreadable spans only count if none can be read.
* Thresholds are team-defined starting values and live in ``RuleConfig``.

The rules are regular expressions over the text, so they are approximate. They are
meant to be read, argued with, and adjusted after the Risk Validation task (#34).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

LEVEL_SCORE = {"Unknown": 0, "Low": 1, "Medium": 2, "High": 3}


@dataclass(frozen=True)
class RuleConfig:
    """Team-defined starting thresholds from RISK_DESIGN.md."""

    min_safe_notice_days: int = 60  # termination notice below this is short
    min_safe_cap_months: int = 12  # a liability cap below this many months of fees is low
    max_short_exclusivity_months: int = 12  # exclusivity up to this long counts as short
    broad_carve_out_count: int = 3  # this many distinct cap exceptions counts as broad
    assume_adverse_when_direction_unknown: bool = True


DEFAULT_CONFIG = RuleConfig()


@dataclass(frozen=True)
class RiskResult:
    category: str
    level: str  # Low / Medium / High / Unknown
    rule_id: str  # e.g. "TFC-H1"; ends in ".default" when no rule could read the clause
    rationale: str
    evidence: str = ""  # the span that drove the result
    facts: dict = field(default_factory=dict)  # what the rules read from the text
    assumptions: tuple[str, ...] = ()
    span_levels: tuple[str, ...] = ()  # level of each span when several were scored

    @property
    def score(self) -> int:
        return LEVEL_SCORE[self.level]

    @property
    def is_default(self) -> bool:
        return self.rule_id.endswith(".default")

    def to_dict(self) -> dict:
        row = asdict(self)
        row["score"] = self.score
        return row


# --- text helpers -------------------------------------------------------------------

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Straighten quotes and collapse whitespace; keeps case, wording is untouched."""
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip()


_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    w: 10 * i
    for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)
}
_NUM_WORD = "|".join(sorted([*_UNITS, *_TENS, "hundred"], key=len, reverse=True))
_NUM = rf"(?:\d+|(?:{_NUM_WORD})(?:[\s-]+(?:{_NUM_WORD}))*)"
_DURATION_RE = re.compile(
    rf"\b(?P<n>{_NUM})[\s-]*(?:\(\s*(?P<p>{_NUM})\s*\)[\s-]*)?"
    r"(?P<unit>business\s+days?|working\s+days?|calendar\s+days?|days?|weeks?|months?|years?)\b",
    re.I,
)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def words_to_int(text: str) -> int | None:
    """'sixty' -> 60, 'one hundred eighty' -> 180, '12' -> 12, anything else -> None."""
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    total, seen = 0, False
    for token in re.split(r"[\s-]+", text):
        if token in _UNITS:
            total += _UNITS[token]
        elif token in _TENS:
            total += _TENS[token]
        elif token == "hundred":
            total = (total or 1) * 100
        else:
            return None
        seen = True
    return total if seen else None


@dataclass(frozen=True)
class Duration:
    days: int
    text: str
    start: int
    end: int


def find_durations(text: str) -> list[Duration]:
    """Every '60 days', 'six (6) months', 'two weeks' in the text, converted to days."""
    found = []
    for match in _DURATION_RE.finditer(text):
        # 'sixty (60) days' states the number twice; either reading is fine.
        values = [words_to_int(match["n"]), words_to_int(match["p"]) if match["p"] else None]
        number = next((v for v in values if v is not None), None)
        if number is None:
            continue
        unit = match["unit"].lower().split()[-1].rstrip("s")
        found.append(Duration(number * _UNIT_DAYS[unit], match[0], match.start(), match.end()))
    return found


def _near(text: str, duration: Duration, cue: re.Pattern, radius: int = 80) -> bool:
    return bool(cue.search(text[max(0, duration.start - radius) : duration.end + radius]))


# --- who does a term apply to? --------------------------------------------------------

_MUTUAL_RE = re.compile(
    r"\b(?:either|each|both|neither|any|no|a)\s+(?:of\s+the\s+)?part(?:y|ies)\b"
    r"|\bthe\s+parties\b|\bmutual(?:ly)?\b|\beither\s+of\s+(?:them|us)\b"
    r"|\beach\s+other\b|\bone\s+another\b"
    r"|\beither\s+(?:\w+\s+){1,3}?or\s+\w+|\beach\s+of\s+(?:\w+\s+){1,4}?and\b",
    re.I,
)
# 'Licensor and Licensee may ...': two capitalised names joined before a verb (case-sensitive).
_PAIR_RE = re.compile(r"\b(?:the\s+)?[A-Z][\w.&-]*\s+(?:and|or)\s+(?:the\s+)?[A-Z][\w.&-]*\s+(?:may|shall|will|can)\b")


def _is_mutual(window: str) -> bool:
    return bool(_MUTUAL_RE.search(window) or _PAIR_RE.search(window))


def _alias_regex(aliases: Sequence[str]) -> re.Pattern | None:
    names = [a.strip() for a in aliases if a and a.strip()]
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.I)


def _subject_of(head: str, size: int) -> str:
    """The words just before a verb, cut at the last comma unless that leaves no name.

    'Upon a Change of Control of Licensee, Licensor may terminate' -> 'Licensor may';
    'Licensee, at its option, may terminate' keeps 'Licensee' because 'at its option'
    holds no capitalised name.
    """
    pieces = head.split(",")
    subject = pieces[-1]
    if len(pieces) > 1 and not re.search(r"\b[A-Z]", subject):
        subject = pieces[-2] + "," + subject
    return subject[-size:]


def direction(
    text: str,
    trigger: re.Pattern,
    aliases: Sequence[str],
    *,
    before: int = 40,
    after: int = 0,
    keep: Callable[[re.Match], bool] | None = None,
) -> str:
    """Which side a term applies to, judged around each match of ``trigger``.

    A reviewer alias counts only if it sits within ``before`` characters ahead of the
    trigger (the grammatical subject) or ``after`` characters behind it, so a party that
    is merely mentioned earlier in the sentence is not mistaken for the actor. 'Either
    party' style wording is looked for in a wider window.

    Returns ``"mutual"`` (a 'either party' cue, or both sides appear), ``"reviewer"``
    (a reviewer alias is in the window), ``"counterparty"`` (aliases were given but none
    is in the window, so the clause names someone else), ``"unknown"`` (no aliases
    given and no mutual cue), or ``"none"`` (the trigger never matched, or ``keep``
    rejected every match).
    """
    alias_re = _alias_regex(aliases)
    sides: set[str] = set()
    for match in trigger.finditer(text):
        if keep is not None and not keep(match):
            continue
        sentence_head = text[max(0, match.start() - 100) : match.start()].rsplit(". ", 1)[-1]
        window = _subject_of(sentence_head, before) + text[match.start() : match.end() + after]
        mutual_window = sentence_head + text[match.start() : match.end() + max(after, 80)]
        if _is_mutual(mutual_window):
            sides.add("mutual")
        elif alias_re is None:
            sides.add("unknown")
        elif alias_re.search(window):
            sides.add("reviewer")
        else:
            sides.add("counterparty")
    if not sides:
        return "none"
    if "mutual" in sides or {"reviewer", "counterparty"} <= sides:
        return "mutual"
    return next(iter(sides))


def _merge_directions(*sides: str) -> str:
    """Combine directions found with different triggers; disagreement means both sides."""
    found = {side for side in sides if side != "none"}
    if not found:
        return "none"
    return next(iter(found)) if len(found) == 1 else "mutual"


# --- rule plumbing ------------------------------------------------------------------


@dataclass
class _Finding:
    level: str
    rule_id: str
    rationale: str
    facts: dict
    assumptions: list[str] = field(default_factory=list)


_Rule = Callable[[str, RuleConfig, Sequence[str]], _Finding]

_ASSUMED = (
    "The clause does not say which party is the reviewer and no reviewer_aliases were "
    "given, so it was read against the reviewer."
)
_NOT_ASSUMED = (
    "Which party the term favours is unknown and assume_adverse_when_direction_unknown "
    "is off, so the direction was not held against the reviewer."
)


def _resolve_adverse(side: str, adverse_sides: Iterable[str], cfg: RuleConfig) -> tuple[bool, list[str]]:
    """Is this term working against the reviewer? Returns (adverse, assumptions)."""
    if side == "unknown":
        return cfg.assume_adverse_when_direction_unknown, [
            _ASSUMED if cfg.assume_adverse_when_direction_unknown else _NOT_ASSUMED
        ]
    return side in set(adverse_sides), []


def _days_text(days: int | None) -> str:
    return "no notice period stated" if days is None else f"{days} days' notice"


# --- Termination for Convenience ----------------------------------------------------

_TERM_VERB_RE = re.compile(r"\b(?:terminat|cancel|cease|discontinu)\w*", re.I)
_BREACH_RE = re.compile(r"breach|default|cure|insolven|bankrupt|fail(?:s|ure)?\s+to", re.I)
_NOTICE_RE = re.compile(r"notice|notif|prior\s+to|in\s+advance|advance", re.I)
_IMMEDIATE_RE = re.compile(
    r"\bimmediate(?:ly)?\b|\bwithout\s+(?:any\s+)?(?:prior\s+)?(?:written\s+)?notice\b|\bforthwith\b", re.I
)
_FEE_RE = re.compile(
    r"termination\s+fee|early\s+termination|(?:pay|payment\s+of)\s+(?:a|the|an)\s+[\w\s-]{0,25}fee"
    r"|wind[- ]?down|transition(?:al)?\s+(?:assistance|period|services|support)",
    re.I,
)


_REDACTED_RE = re.compile(r"\[\s*\*[\s*]*\]")


_CLAUSE_BREAK_RE = re.compile(r"[;,]|\bor\b", re.I)


def _clause_around(text: str, position: int) -> str:
    """The comma/semicolon/'or' delimited piece of ``text`` that contains ``position``."""
    start = max((m.end() for m in _CLAUSE_BREAK_RE.finditer(text, 0, position)), default=0)
    stop = next((m.start() for m in _CLAUSE_BREAK_RE.finditer(text, position)), len(text))
    return text[start:stop]


def _notice_days(text: str) -> int | None:
    """Shortest notice period in the text, ignoring cure periods for breach."""
    days = []
    for d in find_durations(text):
        piece = _clause_around(text, d.start)
        if _NOTICE_RE.search(piece) and not _BREACH_RE.search(piece):
            days.append(d.days)
    if days:
        return min(days)
    return 0 if _IMMEDIATE_RE.search(text) else None


def _rule_termination(text: str, cfg: RuleConfig, aliases: Sequence[str]) -> _Finding:
    holder = direction(text, _TERM_VERB_RE, aliases)
    notice = _notice_days(text)
    fee = bool(_FEE_RE.search(text))
    redacted = notice is None and bool(_REDACTED_RE.search(text))
    limit = cfg.min_safe_notice_days
    facts = {"who_can_terminate": holder, "notice_days": notice, "notice_redacted": redacted, "fee_or_transition": fee}

    if holder == "none":
        return _Finding("Medium", "TFC-M0.default", "No termination right could be read from the clause; review manually.", facts)
    if fee:
        return _Finding("Low", "TFC-L2", "Termination requires a fee or transition help, which cushions the exit.", facts)
    if redacted:
        return _Finding("Medium", "TFC-M4", "The notice period is redacted in the contract text, so it cannot be compared with 60 days.", facts)
    if holder == "mutual":
        if notice is not None and notice >= limit:
            return _Finding("Low", "TFC-L1", f"Either party can terminate, with {notice} days' notice (at least {limit}).", facts)
        return _Finding("Medium", "TFC-M1", f"Either party can terminate, but with {_days_text(notice)} (under {limit} days).", facts)
    if holder == "reviewer":
        return _Finding("Low", "TFC-L3", "Only the reviewing party can terminate for convenience.", facts)

    adverse, assumptions = _resolve_adverse(holder, {"counterparty"}, cfg)
    if not adverse:
        return _Finding("Medium", "TFC-M3", "A one-sided termination right of unknown direction.", facts, assumptions)
    if notice is None or notice < limit:
        return _Finding("High", "TFC-H1", f"Only the other party can terminate, with {_days_text(notice)} (under {limit} days).", facts, assumptions)
    return _Finding("Medium", "TFC-M2", f"Only the other party can terminate, but with {notice} days' notice (at least {limit}).", facts, assumptions)


# --- Cap on Liability ---------------------------------------------------------------

_LIABILITY_RE = re.compile(r"\bliab\w*|\bresponsib\w*", re.I)
_UNLIMITED_RE = re.compile(
    r"\bunlimited\b|\buncapped\b|\bwithout\s+(?:any\s+)?(?:limit|cap)\b"
    r"|\bno\s+(?:limit|cap|maximum)\b",
    re.I,
)
_FEE_CUE_RE = re.compile(
    r"\bfees?\b|amounts?\s+(?:paid|payable)|charges|payments?|royalt|purchase\s+price|consideration|compensation",
    re.I,
)
_DOLLAR_RE = re.compile(r"\$\s?\d|\bUSD\b|\bdollars\b", re.I)
_FEES_UNSPECIFIED_RE = re.compile(
    r"(?:fees|amounts?|charges|payments?|royalties|purchase\s+price|consideration|compensation)\s+"
    r"(?:actually\s+)?(?:paid|payable|received)|total\s+(?:price|fees|amount)|(?:shall|will)\s+not\s+exceed",
    re.I,
)
_CARVE_OUTS = {
    "gross negligence": r"gross\s+negligen",
    "willful misconduct": r"wil+ful\s+misconduct|intentional\s+misconduct",
    "fraud": r"\bfraud",
    "indemnification": r"indemnif",
    "confidentiality": r"confidential",
    "infringement": r"infring|misappropriat",
    "injury or death": r"bodily\s+injury|personal\s+injury|\bdeath\b",
    "data breach": r"data\s+breach|security\s+breach",
}


def _carve_outs(text: str) -> list[str]:
    return [name for name, pattern in _CARVE_OUTS.items() if re.search(pattern, text, re.I)]


def _cap_months(text: str) -> float | None:
    months = [
        d.days / 30
        for d in find_durations(text)
        if _near(text, d, _FEE_CUE_RE, radius=70) and d.days >= 30
    ]
    return min(months) if months else None


def _rule_cap(text: str, cfg: RuleConfig, aliases: Sequence[str]) -> _Finding:
    protected = direction(text, _LIABILITY_RE, aliases)
    months = _cap_months(text)
    has_dollar = bool(_DOLLAR_RE.search(text))
    has_cap = months is not None or has_dollar or bool(_FEES_UNSPECIFIED_RE.search(text))
    unlimited = bool(_UNLIMITED_RE.search(text))
    carve_outs = _carve_outs(text)
    broad = len(carve_outs) >= cfg.broad_carve_out_count
    limit = cfg.min_safe_cap_months
    facts = {
        "cap_protects": protected,
        "cap_months_of_fees": None if months is None else round(months, 1),
        "has_cap_figure": has_cap,
        "unlimited_language": unlimited,
        "carve_outs": carve_outs,
    }

    if unlimited and not has_cap:
        return _Finding("High", "CAP-H1", "Liability is described as unlimited and no cap figure is given.", facts)
    if protected in ("mutual", "reviewer"):
        if months is not None and months < limit:
            return _Finding("Medium", "CAP-M1", f"Cap is about {months:.0f} months of fees, under {limit}.", facts)
        if broad or (unlimited and has_cap):
            why = ", ".join(carve_outs) if carve_outs else "unlimited exceptions"
            return _Finding("Medium", "CAP-M2", f"Cap has broad exceptions ({why}).", facts)
        if months is not None:
            side = "Mutual" if protected == "mutual" else "The reviewer is protected by a"
            return _Finding("Low", "CAP-L1", f"{side} cap of about {months:.0f} months of fees, with standard carve-outs.", facts)
        if has_cap:
            return _Finding("Medium", "CAP-M3", "A cap exists but is not stated as a number of months of fees, so it cannot be compared with 12 months.", facts)
        return _Finding("Medium", "CAP-M0.default", "No cap figure could be read; the clause may only exclude some kinds of damages. Review manually.", facts)

    adverse, assumptions = _resolve_adverse(protected, {"counterparty"}, cfg)
    if protected == "none":
        return _Finding("Medium", "CAP-M0.default", "No liability wording could be read; review manually.", facts)
    if adverse and has_cap:
        return _Finding("High", "CAP-H2", "A cap on liability protects only the other party, leaving the reviewer exposed.", facts, assumptions)
    if adverse:
        return _Finding("Medium", "CAP-M5", "Only the other party is shielded from some damages, but no monetary cap is stated.", facts, assumptions)
    return _Finding("Medium", "CAP-M4", "A one-sided limit on liability of unknown direction.", facts, assumptions)


# --- Change of Control --------------------------------------------------------------

_COC_TERMINATE_RE = re.compile(r"\b(?:terminat|cancel|rescind)\w*", re.I)
_COC_CONSENT_RE = re.compile(r"\bconsent\b|\bapproval\b|\bpermission\b|deemed\s+(?:to\s+be\s+)?(?:an?\s+)?assignment", re.I)
_COC_CUE_RE = re.compile(
    r"change\s+(?:of|in)\s+control|\bcontrol\b|acquir|\bmerg|consolidat|ownership|reorgani[sz]|sale\s+of\s+(?:all|substantially)|transfer\s+of\s+(?:shares|stock|equity)",
    re.I,
)
_COC_NOT_ASSIGNMENT_RE = re.compile(
    r"(?:shall|will)\s+not\s+(?:be\s+)?(?:considered|deemed|constitute)\s+(?:an?\s+)?assignment"
    r"|\bno\s+(?:transfer|change)\b[^.]{0,100}?(?:shall|will)\s+be\s+(?:considered|deemed)\s+(?:an?\s+)?assignment",
    re.I,
)
_COC_NOTICE_RE = re.compile(r"\bnotif|\bnotice\b|\binform", re.I)
_COC_RENEGOTIATE_RE = re.compile(r"renegotiat|re-negotiat|negotiate|good\s+faith|discuss|meet\s+and\s+confer", re.I)


def _rule_change_of_control(text: str, cfg: RuleConfig, aliases: Sequence[str]) -> _Finding:
    cue_spans = [m.span() for m in _COC_CUE_RE.finditer(text)]

    def near_cue(match: re.Match) -> bool:  # the right must sit close to change-of-control wording
        return any(start - 300 <= match.start() <= end + 300 for start, end in cue_spans)

    # Whoever terminates is the subject before the verb; whoever consents is named after it.
    holder = _merge_directions(
        direction(text, _COC_TERMINATE_RE, aliases, keep=near_cue),
        direction(text, _COC_CONSENT_RE, aliases, after=25, keep=near_cue),
    )
    notice = bool(_COC_NOTICE_RE.search(text))
    renegotiate = bool(_COC_RENEGOTIATE_RE.search(text))
    terminates = bool(re.search(r"terminat|cancel|rescind", text, re.I))
    not_assignment = bool(_COC_NOT_ASSIGNMENT_RE.search(text))
    facts = {"termination_or_consent_right_held_by": holder, "terminate_right": terminates, "notice": notice, "renegotiation": renegotiate, "not_an_assignment": not_assignment}

    if not_assignment and not terminates:
        return _Finding("Low", "COC-L3", "A change of control is expressly not treated as an assignment and needs no consent.", facts)
    if holder != "none":
        if not terminates and _REASONABLE_RE.search(text):
            return _Finding("Medium", "COC-M4", "A change of control needs consent that may not be unreasonably withheld.", facts)
        if holder == "reviewer":
            return _Finding("Low", "COC-L2", "A change of control gives the reviewer, not the other party, the right to terminate or consent.", facts)
        adverse, assumptions = _resolve_adverse(holder, {"counterparty", "mutual"}, cfg)
        if adverse:
            return _Finding("High", "COC-H1", "The other party can terminate, or its consent is needed, on a change of control.", facts, assumptions)
        return _Finding("Medium", "COC-M3", "A termination or consent right on change of control, direction unknown.", facts, assumptions)
    if notice and renegotiate:
        return _Finding("Medium", "COC-M1", "A change of control requires notice and opens a right to renegotiate.", facts)
    if notice:
        return _Finding("Low", "COC-L1", "A change of control requires notice only.", facts)
    return _Finding("Medium", "COC-M0.default", "No change-of-control consequence could be read; review manually.", facts)


# --- Anti-Assignment ----------------------------------------------------------------

# Active ('Customer may not assign') and passive ('may not be assigned by Customer') wording.
_ASSIGN_ACTIVE_RE = re.compile(r"\b(?:assign|transfer|delegate|sublicense)\b", re.I)
_ASSIGN_PASSIVE_RE = re.compile(r"\b(?:assigned|transferred|delegated|assignable|transferable)\b[^.;]{0,30}?\bby\b", re.I)
_BAN_RE = re.compile(
    r"(?:may\s+not|shall\s+not|cannot|will\s+not|not\s+be|no\s+(?:party|assignment))\s+(?:be\s+)?(?:assign|transfer|delegat)"
    r"|neither\s+part(?:y|ies)\s+(?:may|shall|will|can)\s+(?:assign|transfer|delegat)|\bnot\s+(?:be\s+)?(?:assignable|transferable)",
    re.I,
)


def _restricted_party(text: str, aliases: Sequence[str]) -> str:
    return _merge_directions(
        direction(text, _ASSIGN_ACTIVE_RE, aliases),
        direction(text, _ASSIGN_PASSIVE_RE, aliases, after=25),
    )
_CONSENT_RE = re.compile(r"\bconsent\b|\bapproval\b|\bpermission\b|\bauthori[sz]ation\b", re.I)
_SOLE_DISCRETION_RE = re.compile(r"sole\s+(?:and\s+absolute\s+)?discretion|absolute\s+discretion", re.I)
_REASONABLE_RE = re.compile(r"unreasonabl|reasonable\s+consent|arbitrarily", re.I)
_FREE_TRANSFER_RE = re.compile(
    r"without\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?(?:consent|approval)|free(?:ly)?\s+(?:assign|transfer)"
    r"|(?:may|right\s+to)\s+assign|not\s+require\s+consent",
    re.I,
)
_SUCCESSOR_RE = re.compile(r"affiliate|successor|merger|merge|consolidat|substantially\s+all|operation\s+of\s+law|acquir", re.I)


def _rule_anti_assignment(text: str, cfg: RuleConfig, aliases: Sequence[str]) -> _Finding:
    restricted = _restricted_party(text, aliases)
    consent = bool(_CONSENT_RE.search(text))
    ban = bool(_BAN_RE.search(text)) and not consent
    sole = bool(_SOLE_DISCRETION_RE.search(text)) and not _REASONABLE_RE.search(text)
    free_transfer = bool(_FREE_TRANSFER_RE.search(text)) and bool(_SUCCESSOR_RE.search(text))
    facts = {
        "restricted_party": restricted,
        "full_ban": ban,
        "consent_required": consent,
        "sole_discretion": sole,
        "free_transfer_to_affiliates_or_successors": free_transfer,
    }

    if restricted == "none":
        return _Finding("Medium", "ASG-M0.default", "No assignment restriction could be read; review manually.", facts)
    if restricted == "counterparty":
        return _Finding("Low", "ASG-L2", "Only the other party is restricted from assigning.", facts)
    if free_transfer and not sole:
        return _Finding("Low", "ASG-L1", "Assignment is free to affiliates or merger successors.", facts)
    adverse, assumptions = _resolve_adverse(restricted, {"reviewer", "mutual"}, cfg)
    if ban:
        return _Finding("High", "ASG-H1", "Assignment is banned outright.", facts, assumptions)
    if sole:
        if restricted == "mutual" or not adverse:
            return _Finding("Medium", "ASG-M2", "Consent is at the other side's sole discretion, but it applies to both parties.", facts, assumptions)
        return _Finding("High", "ASG-H2", "The reviewer needs consent that can be withheld at the other party's sole discretion.", facts, assumptions)
    if consent and _REASONABLE_RE.search(text):
        return _Finding("Medium", "ASG-M1", "Assignment needs consent that may not be unreasonably withheld.", facts, assumptions)
    if consent:
        return _Finding("Medium", "ASG-M3", "Assignment needs consent; the clause does not say it cannot be withheld freely.", facts, assumptions)
    return _Finding("Medium", "ASG-M0.default", "An assignment restriction was found but its terms could not be read; review manually.", facts, assumptions)


# --- Exclusivity --------------------------------------------------------------------

_EXCLUSIVITY_WORD_RE = re.compile(r"exclusiv|\bsole\s+(?:and\s+exclusive\s+)?(?:provider|source|supplier|vendor)|\bsolely\b", re.I)
_GRANT_EXCLUSIVE_RE = re.compile(r"\b(?:grant|appoint|designat)(?:s|ed)?\s+(?:to\s+)?(?P<grantee>.{0,80}?)\bexclusive", re.I | re.S)
_RESTRICT_RE = re.compile(
    r"agrees?\s+(?:not\s+)?to|agrees?\s+that|shall\s+not|will\s+not|\bonly\s+(?:from|with|through)|sole\s+and\s+exclusive\s+(?:provider|supplier|source|vendor)",
    re.I,
)
_PERFORMANCE_RE = re.compile(r"minimum|perform|quota|milestone|forfeit|revert|lose|convert", re.I)
_EXIT_RE = re.compile(r"terminat|expire|non-?exclusive|convert|revert|upon\s+notice", re.I)
_WORLDWIDE_RE = re.compile(r"worldwide|throughout\s+the\s+world|global", re.I)
_TERRITORY_RE = re.compile(r"territory|region|country|\bin\s+the\s+(?:United\s+States|U\.S\.)|within\s+[A-Z]", re.I)


def _exclusivity_bound_party(text: str, aliases: Sequence[str]) -> str:
    """Whose hands are tied: ``reviewer``, ``counterparty``, ``mutual`` or ``unknown``."""
    grant = _GRANT_EXCLUSIVE_RE.search(text)
    if grant:
        alias_re = _alias_regex(aliases)
        if alias_re is None:
            return "unknown"
        # The grantee benefits; the party that granted exclusivity is the one tied.
        return "counterparty" if alias_re.search(grant["grantee"]) else "reviewer"
    side = direction(text, _RESTRICT_RE, aliases)
    if side == "none" and _EXCLUSIVITY_WORD_RE.search(text):
        return "mutual" if _is_mutual(text) else "unknown"
    return side


def _rule_exclusivity(text: str, cfg: RuleConfig, aliases: Sequence[str]) -> _Finding:
    bound = _exclusivity_bound_party(text, aliases)
    terms = [d.days / 30 for d in find_durations(text) if d.days >= 30]
    term_months = max(terms) if terms else None
    limited_territory = bool(_TERRITORY_RE.search(text)) and not _WORLDWIDE_RE.search(text)
    conditions = bool(_PERFORMANCE_RE.search(text))
    exit_route = bool(_EXIT_RE.search(text))
    short = term_months is not None and term_months <= cfg.max_short_exclusivity_months
    facts = {
        "bound_party": bound,
        "term_months": None if term_months is None else round(term_months, 1),
        "limited_territory": limited_territory,
        "minimums_or_performance_conditions": conditions,
        "exit_route": exit_route,
    }

    if bound == "none":
        return _Finding("Medium", "EXC-M0.default", "No exclusivity obligation could be read; review manually.", facts)
    if bound == "counterparty":
        return _Finding("Low", "EXC-L1", "Exclusivity binds the other party, not the reviewer.", facts)
    adverse, assumptions = _resolve_adverse(bound, {"reviewer", "mutual"}, cfg)
    if not adverse:
        return _Finding("Medium", "EXC-M3", "An exclusivity term of unknown direction.", facts, assumptions)
    if short and limited_territory:
        return _Finding("Low", "EXC-L2", f"Exclusivity is short ({term_months:.0f} months) and limited in territory.", facts, assumptions)
    if not conditions and not exit_route:
        return _Finding("High", "EXC-H1", "The reviewer is locked in with no minimums, performance conditions or exit.", facts, assumptions)
    return _Finding("Medium", "EXC-M1", "Exclusivity binds the reviewer but has performance conditions or an exit.", facts, assumptions)


# --- IP Ownership Assignment --------------------------------------------------------

_ASSIGNEE_RE = re.compile(
    r"\b(?:assign|transfer|convey|vest)\w*\b[^.;]{0,200}?\b(?:to|unto)\s+(?:and\s+to\s+)?(?P<who>[^.;]{1,50})", re.I
)
_OWNER_RE = re.compile(
    r"(?P<who>[^.;]{1,60}?)\b(?:shall|will|is|are|be)\s+(?:the\s+)?(?:sole\s+|exclusive\s+|joint\s+)*(?:and\s+)?own(?:er)?s?\b", re.I
)
_ALL_IP_RE = re.compile(
    r"all\s+(?:of\s+)?(?:its\s+|their\s+)?(?:right|title|interest)|all\s+intellectual\s+property|(?:sole\s+and\s+)?exclusive\s+owner|own\s+all",
    re.I,
)
_EXISTING_IP_RE = re.compile(r"pre-?existing|background|existing\s+(?:ip|intellectual|technology|materials)|previously\s+(?:developed|owned)", re.I)
_RESERVATION_RE = re.compile(r"\bretain|\breserv|\bremain|\bexclud|\bexcept|\bnot\s+include|shall\s+not\s+apply|license\s+back|grants?\s+[^.;]{0,60}\blicen[sc]e", re.I)


def _ip_assignee(text: str, aliases: Sequence[str]) -> str:
    alias_re = _alias_regex(aliases)
    sides: set[str] = set()
    for pattern in (_ASSIGNEE_RE, _OWNER_RE):
        for match in pattern.finditer(text):
            who = match["who"]
            if _is_mutual(who):
                sides.add("mutual")
            elif alias_re is None:
                sides.add("unknown")
            else:
                sides.add("reviewer" if alias_re.search(who) else "counterparty")
    if not sides:
        return "none"
    if "mutual" in sides or {"reviewer", "counterparty"} <= sides:
        return "mutual"
    return next(iter(sides))


def _rule_ip_assignment(text: str, cfg: RuleConfig, aliases: Sequence[str]) -> _Finding:
    assignee = _ip_assignee(text, aliases)
    all_ip = bool(_ALL_IP_RE.search(text))
    existing = bool(_EXISTING_IP_RE.search(text))
    reservation = bool(_RESERVATION_RE.search(text))
    facts = {"owner_after_assignment": assignee, "assigns_all_rights": all_ip, "mentions_existing_ip": existing, "reservation_or_license_back": reservation}

    if assignee == "none":
        return _Finding("Medium", "IP-M0.default", "No ownership transfer could be read; review manually.", facts)
    if assignee == "reviewer":
        return _Finding("Low", "IP-L1", "The reviewer owns the work product.", facts)
    adverse, assumptions = _resolve_adverse(assignee, {"counterparty"}, cfg)
    if assignee == "mutual" or not adverse:
        return _Finding("Medium", "IP-M3", "Ownership is shared or its direction is unknown.", facts, assumptions)
    if existing and all_ip and not reservation:
        return _Finding("High", "IP-H1", "The other party owns all work product, including existing IP of the reviewer.", facts, assumptions)
    if all_ip and not reservation:
        return _Finding("High", "IP-H2", "The other party owns all rights with no reservation of the reviewer's existing IP.", facts, assumptions)
    if reservation:
        return _Finding("Medium", "IP-M1", "The other party owns the deliverables, but the reviewer keeps background IP or gets a license.", facts, assumptions)
    return _Finding("Medium", "IP-M2", "The other party owns some work product; scope of the transfer is unclear.", facts, assumptions)


# --- public API ---------------------------------------------------------------------

RULES: dict[str, _Rule] = {
    "Cap on Liability": _rule_cap,
    "Termination for Convenience": _rule_termination,
    "Change of Control": _rule_change_of_control,
    "Anti-Assignment": _rule_anti_assignment,
    "Exclusivity": _rule_exclusivity,
    "IP Ownership Assignment": _rule_ip_assignment,
}


def _score_span(category: str, text: str, cfg: RuleConfig, aliases: Sequence[str]) -> RiskResult:
    finding = RULES[category](normalize(text), cfg, aliases)
    return RiskResult(
        category=category,
        level=finding.level,
        rule_id=finding.rule_id,
        rationale=finding.rationale,
        evidence=text.strip(),
        facts=finding.facts,
        assumptions=tuple(finding.assumptions),
    )


def score_clause(
    category: str,
    evidence: str | Sequence[str] | None,
    reviewer_aliases: Sequence[str] = (),
    config: RuleConfig = DEFAULT_CONFIG,
) -> RiskResult:
    """Score one detected clause category.

    ``evidence`` is the clause text, or a list of spans when the category was found in
    several places. ``None`` or empty means the category was not detected (Unknown).
    """
    spans = [evidence] if isinstance(evidence, str) else list(evidence or [])
    spans = [s for s in spans if s and s.strip()]
    if category not in RULES:
        return RiskResult(category, "Unknown", "NO-RULE", f"Risk Rules v1 has no rule for '{category}'.")
    if not spans:
        return RiskResult(
            category,
            "Unknown",
            "NOT-DETECTED",
            "Clause not detected. Unknown, not Low, because the classifier misses some clauses.",
        )

    results = [_score_span(category, s, config, tuple(reviewer_aliases)) for s in spans]
    readable = [r for r in results if not r.is_default] or results
    best = max(readable, key=lambda r: LEVEL_SCORE[r.level])  # max keeps the first of equal levels
    return RiskResult(
        category=best.category,
        level=best.level,
        rule_id=best.rule_id,
        rationale=best.rationale,
        evidence=best.evidence,
        facts=best.facts,
        assumptions=best.assumptions,
        span_levels=tuple(r.level for r in results) if len(results) > 1 else (),
    )


def score_register(
    detections: Mapping[str, str | Sequence[str] | None],
    reviewer_aliases: Sequence[str] = (),
    config: RuleConfig = DEFAULT_CONFIG,
) -> list[RiskResult]:
    """Score every category with a rule. Categories missing from ``detections`` are Unknown."""
    return [
        score_clause(category, detections.get(category), reviewer_aliases, config)
        for category in RULES
    ]
