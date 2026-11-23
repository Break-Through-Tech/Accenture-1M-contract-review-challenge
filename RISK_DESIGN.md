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