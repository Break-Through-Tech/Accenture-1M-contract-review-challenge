"""Tests for Risk Rules v1 (scripts/risk_rules.py), one block per RISK_DESIGN.md category."""
import pytest

from risk_rules import (
    RULES,
    RuleConfig,
    find_durations,
    score_clause,
    score_register,
    words_to_int,
)


def level(category, text, aliases=(), config=None):
    kwargs = {} if config is None else {"config": config}
    return score_clause(category, text, aliases, **kwargs)


# --- helpers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [("sixty", 60), ("one hundred eighty", 180), ("twenty-four", 24), ("12", 12), ("banana", None)],
)
def test_words_to_int(text, expected):
    assert words_to_int(text) == expected


@pytest.mark.parametrize(
    "text, days",
    [
        ("upon sixty (60) days' notice", 60),
        ("by 6 (six) months prior notice", 180),
        ("two (2) weeks' notice", 14),
        ("a 30-day period", 30),
        ("one hundred eighty (180) days", 180),
        ("for 1 year", 365),
        ("ten (10) business days", 10),
    ],
)
def test_find_durations(text, days):
    assert [d.days for d in find_durations(text)] == [days]


def test_find_durations_ignores_non_durations():
    assert find_durations("Section 2.1(b) and the Effective Date") == []


# --- unknown / not detected ---------------------------------------------------------


@pytest.mark.parametrize("evidence", [None, "", "   ", []])
def test_undetected_category_is_unknown_not_low(evidence):
    result = score_clause("Cap on Liability", evidence)
    assert result.level == "Unknown"
    assert result.score == 0
    assert result.rule_id == "NOT-DETECTED"


def test_category_without_rule_is_unknown():
    assert score_clause("Governing Law", "This Agreement is governed by Delaware law.").rule_id == "NO-RULE"


def test_score_register_covers_every_rule_category():
    register = score_register({"Exclusivity": "Customer agrees to buy exclusively from Supplier."})
    assert [r.category for r in register] == list(RULES)
    assert {r.category: r.level for r in register}["Cap on Liability"] == "Unknown"


# --- Termination for Convenience ----------------------------------------------------


def test_tfc_mutual_long_notice_is_low():
    r = level("Termination for Convenience", "Either party may terminate this Agreement for convenience upon sixty (60) days' prior written notice.")
    assert (r.level, r.rule_id) == ("Low", "TFC-L1")


def test_tfc_mutual_short_notice_is_medium():
    r = level("Termination for Convenience", "Either party may terminate this Agreement upon thirty (30) days' written notice.")
    assert (r.level, r.rule_id) == ("Medium", "TFC-M1")


def test_tfc_only_other_party_short_notice_is_high():
    r = level("Termination for Convenience", "Licensor may terminate this Agreement at any time upon fourteen (14) days' written notice to Licensee.", ("Licensee",))
    assert (r.level, r.rule_id) == ("High", "TFC-H1")
    assert r.assumptions == ()  # direction was known, nothing assumed


def test_tfc_only_other_party_long_notice_is_medium():
    r = level("Termination for Convenience", "Licensor may terminate this Agreement upon ninety (90) days' written notice.", ("Licensee",))
    assert (r.level, r.rule_id) == ("Medium", "TFC-M2")


def test_tfc_only_reviewer_can_terminate_is_low():
    r = level("Termination for Convenience", "Licensee may terminate this Agreement at any time upon ten (10) days' notice.", ("Licensee",))
    assert r.level == "Low"


def test_tfc_fee_or_transition_help_is_low():
    r = level("Termination for Convenience", "Supplier may terminate on 10 days' notice, subject to payment of an early termination fee.", ("Customer",))
    assert (r.level, r.rule_id) == ("Low", "TFC-L2")


def test_tfc_cure_period_is_not_read_as_notice():
    text = "Either party may terminate for convenience on ninety (90) days' notice, or upon a breach not cured within ten (10) days."
    r = level("Termination for Convenience", text)
    assert r.facts["notice_days"] == 90 and r.level == "Low"


def test_tfc_redacted_notice_is_not_called_short():
    r = level("Termination for Convenience", "Licensor may terminate this Agreement by giving [***] days' prior written notice.", ("Licensee",))
    assert (r.level, r.rule_id) == ("Medium", "TFC-M4")


def test_tfc_no_notice_stated_counts_as_short():
    r = level("Termination for Convenience", "Licensor may terminate this Agreement at any time.", ("Licensee",))
    assert r.level == "High"


def test_tfc_unknown_direction_is_read_adversely_and_says_so():
    r = level("Termination for Convenience", "Supplier may terminate this Agreement upon ten (10) days' notice.")
    assert r.level == "High"
    assert r.assumptions


def test_tfc_unknown_direction_can_be_left_neutral():
    cfg = RuleConfig(assume_adverse_when_direction_unknown=False)
    r = level("Termination for Convenience", "Supplier may terminate this Agreement upon ten (10) days' notice.", config=cfg)
    assert (r.level, r.rule_id) == ("Medium", "TFC-M3")


def test_tfc_threshold_is_configurable():
    text = "Either party may terminate this Agreement upon thirty (30) days' notice."
    assert level("Termination for Convenience", text).level == "Medium"
    assert level("Termination for Convenience", text, config=RuleConfig(min_safe_notice_days=30)).level == "Low"


# --- Cap on Liability ---------------------------------------------------------------


def test_cap_mutual_twelve_months_with_standard_carve_outs_is_low():
    text = "IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY EXCEED THE FEES PAID IN THE PRECEDING TWELVE (12) MONTHS, EXCEPT FOR GROSS NEGLIGENCE OR WILLFUL MISCONDUCT."
    r = level("Cap on Liability", text)
    assert (r.level, r.rule_id) == ("Low", "CAP-L1")


def test_cap_mutual_under_twelve_months_is_medium():
    text = "Neither party's total liability shall exceed the fees paid in the preceding six (6) months."
    r = level("Cap on Liability", text)
    assert (r.level, r.rule_id) == ("Medium", "CAP-M1")


def test_cap_broad_exceptions_is_medium():
    text = ("Each party's aggregate liability shall not exceed the fees paid in the prior twelve (12) months, except for gross negligence, "
            "fraud, indemnification obligations and breach of confidentiality.")
    r = level("Cap on Liability", text)
    assert (r.level, r.rule_id) == ("Medium", "CAP-M2")


def test_cap_unlimited_is_high():
    r = level("Cap on Liability", "Supplier's liability under this Agreement shall be unlimited.")
    assert (r.level, r.rule_id) == ("High", "CAP-H1")


def test_cap_without_limitation_boilerplate_is_not_unlimited():
    r = level("Cap on Liability", "Neither party is liable for indirect damages, including, without limitation, lost profits.")
    assert r.facts["unlimited_language"] is False


def test_cap_only_protecting_other_party_is_high():
    r = level("Cap on Liability", "Supplier's total liability shall not exceed $50,000.", ("Customer",))
    assert (r.level, r.rule_id) == ("High", "CAP-H2")


def test_cap_one_sided_exclusion_without_cap_figure_is_medium():
    r = level("Cap on Liability", "In no event shall Supplier be liable for indirect or consequential damages.", ("Customer",))
    assert r.level == "Medium"


def test_cap_unreadable_clause_is_medium_default_never_low():
    r = level("Cap on Liability", "Neither party will be liable for any consequential damages, lost profits or lost savings.")
    assert r.level == "Medium" and r.is_default


# --- Change of Control --------------------------------------------------------------


def test_coc_other_party_can_terminate_is_high():
    r = level("Change of Control", "Upon a Change of Control of Licensee, Licensor may terminate this Agreement upon written notice.", ("Licensee",))
    assert (r.level, r.rule_id) == ("High", "COC-H1")


def test_coc_consent_not_unreasonably_withheld_is_medium():
    r = level("Change of Control", "Any Change of Control of Licensee requires Licensor's prior written consent, not to be unreasonably withheld.", ("Licensee",))
    assert (r.level, r.rule_id) == ("Medium", "COC-M4")


def test_coc_notice_and_renegotiation_is_medium():
    r = level("Change of Control", "Supplier shall notify Customer of any change of control, and the parties shall negotiate in good faith any necessary amendments.")
    assert (r.level, r.rule_id) == ("Medium", "COC-M1")


def test_coc_notice_only_is_low():
    r = level("Change of Control", "Supplier shall give Customer written notice within fifteen (15) days of any Change of Control.")
    assert (r.level, r.rule_id) == ("Low", "COC-L1")


def test_coc_not_treated_as_assignment_is_low():
    r = level("Change of Control", "No change in Control of a party shall be considered an assignment for purposes of this Section.")
    assert (r.level, r.rule_id) == ("Low", "COC-L3")


def test_coc_deemed_assignment_with_carve_out_is_not_called_low():
    text = ("A change of control shall be deemed an assignment requiring consent, provided that a transfer to a common parent "
            "shall not require consent of the other party.")
    assert level("Change of Control", text).level == "High"


# --- Anti-Assignment ----------------------------------------------------------------


def test_assignment_full_ban_is_high():
    r = level("Anti-Assignment", "Neither party may assign or transfer this Agreement.")
    assert (r.level, r.rule_id) == ("High", "ASG-H1")


def test_assignment_sole_discretion_against_reviewer_is_high():
    r = level("Anti-Assignment", "Customer may not assign this Agreement without Supplier's consent, which may be withheld in Supplier's sole discretion.", ("Customer",))
    assert (r.level, r.rule_id) == ("High", "ASG-H2")


def test_assignment_sole_discretion_mutual_is_medium():
    r = level("Anti-Assignment", "Neither party may assign this Agreement without the other's consent, which may be withheld in its sole discretion.")
    assert (r.level, r.rule_id) == ("Medium", "ASG-M2")


def test_assignment_reasonable_consent_is_medium():
    r = level("Anti-Assignment", "Neither party may assign this Agreement without the other party's consent, such consent not to be unreasonably withheld.")
    assert (r.level, r.rule_id) == ("Medium", "ASG-M1")


def test_assignment_consent_without_reasonableness_wording_is_medium():
    r = level("Anti-Assignment", "Neither party may assign this Agreement without the other party's prior written consent.")
    assert (r.level, r.rule_id) == ("Medium", "ASG-M3")


def test_assignment_free_to_affiliates_and_successors_is_low():
    r = level("Anti-Assignment", "Either party may assign this Agreement without consent to an affiliate or to a successor in a merger.")
    assert (r.level, r.rule_id) == ("Low", "ASG-L1")


def test_assignment_only_other_party_restricted_is_low():
    r = level("Anti-Assignment", "Supplier may not assign this Agreement without Customer's consent.", ("Customer",))
    assert (r.level, r.rule_id) == ("Low", "ASG-L2")


def test_assignment_passive_wording_names_restricted_party():
    r = level("Anti-Assignment", "This Agreement is not assignable by Distributor without prior written consent of Tripath.", ("Distributor",))
    assert r.facts["restricted_party"] == "reviewer"


# --- Exclusivity --------------------------------------------------------------------


def test_exclusivity_lock_in_without_minimums_or_exit_is_high():
    r = level("Exclusivity", "Customer agrees that Supplier shall be its sole and exclusive provider of the Services during the Term.", ("Customer",))
    assert (r.level, r.rule_id) == ("High", "EXC-H1")


def test_exclusivity_binding_the_other_party_is_low():
    r = level("Exclusivity", "Licensor grants Licensee an exclusive license in the Territory.", ("Licensee",))
    assert (r.level, r.rule_id) == ("Low", "EXC-L1")


def test_exclusivity_short_and_narrow_is_low():
    r = level("Exclusivity", "Supplier shall not appoint another distributor in the Territory for six (6) months.", ("Supplier",))
    assert (r.level, r.rule_id) == ("Low", "EXC-L2")


def test_exclusivity_with_performance_conditions_and_exit_is_medium():
    text = ("Customer agrees to purchase exclusively from Supplier, provided Supplier meets the minimum service levels, "
            "until terminated on ninety (90) days' notice.")
    r = level("Exclusivity", text, ("Customer",))
    assert (r.level, r.rule_id) == ("Medium", "EXC-M1")


# --- IP Ownership Assignment --------------------------------------------------------


def test_ip_reviewer_owns_work_product_is_low():
    r = level("IP Ownership Assignment", "Supplier hereby assigns to Customer all right, title and interest in the Work Product.", ("Customer",))
    assert (r.level, r.rule_id) == ("Low", "IP-L1")


def test_ip_other_party_owns_everything_including_existing_ip_is_high():
    text = "Consultant hereby assigns to Company all right, title and interest in all work product, including any pre-existing materials of Consultant."
    r = level("IP Ownership Assignment", text, ("Consultant",))
    assert (r.level, r.rule_id) == ("High", "IP-H1")


def test_ip_other_party_owns_deliverables_reviewer_keeps_background_is_medium():
    text = "Vendor assigns to Customer all rights in the deliverables; Vendor retains all rights in its pre-existing IP and grants Customer a license to it."
    r = level("IP Ownership Assignment", text, ("Vendor",))
    assert (r.level, r.rule_id) == ("Medium", "IP-M1")


# --- several spans ------------------------------------------------------------------


def test_highest_level_among_spans_wins():
    spans = [
        "Either party may terminate this Agreement upon sixty (60) days' notice.",
        "Licensor may terminate this Agreement upon five (5) days' notice.",
    ]
    r = score_clause("Termination for Convenience", spans, ("Licensee",))
    assert r.level == "High"
    assert r.span_levels == ("Low", "High")
    assert "five (5) days" in r.evidence


def test_unreadable_span_does_not_outrank_a_readable_one():
    spans = ["The parties shall cooperate in good faith.", "Either party may terminate this Agreement upon sixty (60) days' notice."]
    assert score_clause("Termination for Convenience", spans).level == "Low"


def test_all_spans_unreadable_falls_back_to_medium_default():
    r = score_clause("Termination for Convenience", ["The parties shall cooperate in good faith."])
    assert r.level == "Medium" and r.is_default


def test_to_dict_is_flat_enough_for_a_dataframe():
    row = level("Exclusivity", "Customer agrees to buy exclusively from Supplier.", ("Customer",)).to_dict()
    assert {"category", "level", "score", "rule_id", "rationale", "evidence", "facts", "assumptions"} <= set(row)
