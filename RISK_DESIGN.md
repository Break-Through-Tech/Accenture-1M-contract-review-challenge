# Risk Design: Risk-Factor Table

Issue #30. This table defines what makes a detected clause Low, Medium, or High risk for six clause categories. The rule-based scoring layer (Risk Rules v1) is built from it.

## Assumptions
- Risk is measured from the point of view of the party reviewing the contract (the client).
- Thresholds such as days of notice and months of fees are team-defined starting values.
- A category that is not detected scores Unknown, not Low, because the model misses some clauses (for example, the baseline recall for Change of Control is 0.24).
- If a category appears in several places in a contract, the highest risk applies.
- Scope is six categories. The remaining CUAD categories can be added using the same format.

## Risk-factor table

| Category | High risk | Medium risk | Low risk | Why it matters |
|---|---|---|---|---|
| Cap on Liability | Liability is unlimited for us, or their cap is very low while ours is uncapped | Mutual cap under 12 months of fees, or broad exceptions | Mutual cap of 12+ months of fees with standard carve-outs | Decides how much money we can lose if something goes wrong |
| Termination for Convenience | Only the other party can terminate, with under 60 days' notice | Mutual with under 60 days' notice, or one-sided with 60+ days | Mutual with 60+ days' notice, or a fee or transition help is required | Affects whether the contract's value can be pulled out from under us |
| Change of Control | Other party can terminate or needs consent if we are acquired | Notice plus a right to renegotiate | Notice only | Can block a sale of the company or cancel a key contract |
| Anti-Assignment | Consent at sole discretion for us only, or a full ban | Consent required but not unreasonably withheld | Free transfer to affiliates and merger successors | Limits our ability to restructure or sell |
| Exclusivity | We are locked in with no minimums or exit | Term over 1 year or limited territory, with performance conditions | Short and narrow, or binds the other party | Limits other deals and can trap us in a weak partnership |
| IP Ownership Assignment | Other party owns all work product, including our existing IP | Other party owns deliverables but we keep our background IP and get a license | We own the work product | Decides who owns the valuable work |

## Validation
The thresholds above will be checked against advisor-reviewed clause examples in the Risk Validation task, and adjusted based on what we learn.

## Implementation: Risk Rules v1
Issue #31. `scripts/risk_rules.py` turns the table above into Python rules; `python scripts/run_risk_rules.py` writes sample outputs to `results/risk_rules/`. Each result carries the level, a rule id, a one-sentence rationale, the facts read from the text, and any assumptions made.

| Category | Rule ids (High / Medium / Low) |
|---|---|
| Cap on Liability | CAP-H1 unlimited, CAP-H2 cap shields only the other party / CAP-M1 under 12 months, CAP-M2 broad exceptions, CAP-M3 cap not in months of fees, CAP-M4/M5 one-sided / CAP-L1 mutual 12+ months |
| Termination for Convenience | TFC-H1 only the other party, under 60 days / TFC-M1 mutual under 60 days, TFC-M2 one-sided 60+ days, TFC-M4 notice redacted / TFC-L1 mutual 60+ days, TFC-L2 fee or transition help, TFC-L3 only we can terminate |
| Change of Control | COC-H1 other party terminates or must consent / COC-M1 notice plus renegotiation, COC-M4 consent not unreasonably withheld / COC-L1 notice only, COC-L2 we hold the right, COC-L3 not treated as an assignment |
| Anti-Assignment | ASG-H1 full ban, ASG-H2 sole discretion against us / ASG-M1 reasonable consent, ASG-M2 sole discretion but mutual, ASG-M3 consent, silent on reasonableness / ASG-L1 free to affiliates and successors, ASG-L2 only the other party restricted |
| Exclusivity | EXC-H1 locked in, no minimums or exit / EXC-M1 conditions or exit exist / EXC-L1 binds the other party, EXC-L2 short and narrow |
| IP Ownership Assignment | IP-H1 other party owns all incl. our existing IP, IP-H2 owns all, no reservation / IP-M1 owns deliverables, we keep background IP, IP-M2 scope unclear / IP-L1 we own it |

Choices the table did not settle:
- **Which party is "us".** Clause text does not say. Pass `reviewer_aliases` (party names or roles); without them a one-sided term is read against the reviewer, and the result lists that assumption. `RuleConfig(assume_adverse_when_direction_unknown=False)` turns that off. The end-to-end pipeline (#36) needs to supply the reviewer's party name.
- **Low needs positive evidence.** A detected clause the rules cannot read is Medium with a rule id ending `.default`. A category that is not detected is Unknown (score 0), as above.
- **Several spans in one category:** the highest level among the readable spans wins; `span_levels` shows all of them.
- **Thresholds** (60 days, 12 months, 3 exceptions) are fields of `RuleConfig`, so Risk Validation (#34) can change them without touching the rules.
- **Limits.** Rules are regular expressions over the clause text. They see only the span they are given, so a minimum-purchase or exit clause written elsewhere in the contract is invisible to them (this makes Exclusivity lean High). They have not been checked against advisor labels yet.
