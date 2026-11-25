# Risk Rules v1: sample outputs

Rules applied to the annotated spans of the **val** contracts (72 contracts, 216 contract-category rows); no reviewer aliases given (direction of one-sided terms is unknown). These are gold spans, so the table shows rule behaviour on correctly detected clauses, not end-to-end quality. No advisor labels exist yet; agreement with them is the job of the Risk Validation task (#34).

## Level counts by category

`default_rule` counts rows where no rule could read the clause and Medium is the fallback.

| category | High | Medium | Low | default_rule |
|---|---|---|---|---|
| Cap on Liability | 9 | 39 | 1 | 16 |
| Termination for Convenience | 10 | 15 | 9 | 0 |
| Change of Control | 12 | 4 | 3 | 3 |
| Anti-Assignment | 6 | 42 | 16 | 3 |
| Exclusivity | 28 | 1 | 1 | 0 |
| IP Ownership Assignment | 14 | 6 | 0 | 1 |

## Worked examples

### Cap on Liability

**High** (`CAP-H2`, FUSIONPHARMACEUTICALSINC_06_05_2020-EX-10.17-Supply Agreement - FUSION)
> CPDC'S MAXIMUM LIABILITY TO FUSION UNDER THIS SUPPLY AGREEMENT FOR ANY REASON WHATSOEVER, INCLUDING, WILL NOT EXCEED [***].
Rationale: A cap on liability protects only the other party, leaving the reviewer exposed.
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.

**Medium** (`CAP-M5`, DOMINIADVISORTRUST_02_18_2005-EX-99.(H)(2)-SPONSORSHIP AGREEMENT)
> Limitation of Liability of the Sponsor.
Rationale: Only the other party is shielded from some damages, but no monetary cap is stated.
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.

**Low** (`CAP-L1`, GopageCorp_20140221_10-K_EX-10.1_8432966_EX-10.1_Content License Agreement)
> EACH PARTY'S LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT WILL NOT EXCEED AN AMOUNT EQUAL TO THE AGGREGATE AMOUNTS PAID OR PAYABLE TO LICENSOR IN THE TWELVE (12) MONTHS PRECEDING THE COMMENCEMENT OF THE CLAIM.
Rationale: Mutual cap of about 12 months of fees, with standard carve-outs.

### Termination for Convenience

**High** (`TFC-H1`, SENMIAOTECHNOLOGYLTD_02_19_2019-EX-10.5-Collaboration Agreement)
> Party A is entitled to unilaterally terminate this Agreement within three natural months from the signing date of this Agreement.
Rationale: Only the other party can terminate, with no notice period stated (under 60 days).
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.

**Medium** (`TFC-M1`, KIROMICBIOPHARMA,INC_05_11_2020-EX-10.23-CONSULTING AGREEMENT)
> Either Consultant or Company may terminate this Agreement upon prior written notice thereof to the other party.
Rationale: Either party can terminate, but with no notice period stated (under 60 days).

**Low** (`TFC-L1`, ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT)
> Each of the Recipient and the Provider may, in their sole discretion, terminate this Agreement in whole or in part, at any time without cause, and without liability except, in the case of the Recipient, for required payment for services rendered and reimbursement for authorized expenses incurred, by providing at least 90 (ninety) days' prior written notice to the other party (such date, the "Services Termination Date...
Rationale: Either party can terminate, with 90 days' notice (at least 60).

### Change of Control

**High** (`COC-H1`, ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply Agreement)
> A change of control shall be deemed an assignment requiring consent hereunder provided that any transfer or assignment that results in Seller's and Buyer's current common parent, Reynolds Group Holdings Limited, ceasing to control either party shall not require consent of the other party.
Rationale: The other party can terminate, or its consent is needed, on a change of control.

**Medium** (`COC-M4`, AzulSa_20170303_F-1A_EX-10.3_9943903_EX-10.3_Maintenance Agreement1)
> Consequently either this Agreement or any of the respective rights or obligations of the Parties hereunder may be assigned or otherwise transferred, in whole or in part, in any form whatsoever (including by way of change of Control), by either Party subject to the prior written consent of the other Party, which consent shall not be unreasonably withheld or delayed, and any attempt to do so without such consent shall ...
Rationale: A change of control needs consent that may not be unreasonably withheld.

**Low** (`COC-L1`, IPAYMENT,INC_05_14_2007-EX-10.1-SPONSORSHIP AGREEMENT)
> In addition, ISO shall provide written notice to SERVICERS within three (3) Business Days of any Change of Control of ISO.
Rationale: A change of control requires notice only.

### Anti-Assignment

**High** (`ASG-H1`, DigitalCinemaDestinationsCorp_20111220_S-1_EX-10.10_7346719_EX-10.10_Affiliate Agreement)
> This Agreement shall not be assignable by either party unless the assignee expressly assumes in writing the obligations of the assignor hereunder.
Rationale: Assignment is banned outright.

**Medium** (`ASG-M3`, KIROMICBIOPHARMA,INC_05_11_2020-EX-10.23-CONSULTING AGREEMENT)
> Neither this Agreement nor any right hereunder or interest herein may be assigned or transferred by Consultant without the express written consent of Company.
Rationale: Assignment needs consent; the clause does not say it cannot be withheld freely.
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.

**Low** (`ASG-L1`, Antares Pharma, Inc. - Manufacturing Agreement)
> Except as otherwise provided in this Section 12.2, neither this Agreement nor any interest hereunder shall be assignable by any Party without the prior written consent of the other (which consent shall not be unreasonably withheld, conditioned or delayed); provided, however, that either Party may assign this Agreement to any wholly-owned subsidiary or to any successor by merger or sale of substantially all of its bus...
Rationale: Assignment is free to affiliates or merger successors.

### Exclusivity

**High** (`EXC-H1`, Antares Pharma, Inc. - Manufacturing Agreement)
> Starting on the Effective Date, Antares or its Subcontractor shall provide the Manufacturing Services in order to manufacture Devices, Products, sample Products and Trainers exclusively for AMAG for the Territory, all in accordance with the Specifications, Applicable Laws, Quality Agreement and this Agreement.
Rationale: The reviewer is locked in with no minimums, performance conditions or exit.
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.

**Medium** (`EXC-M1`, GopageCorp_20140221_10-K_EX-10.1_8432966_EX-10.1_Content License Agreement)
> Subject to Licensee's ongoing compliance with Section 3.2 and all other terms and conditions of this Agreement, Licensor grants to Licensee an exclusive (save for rights reserved to Licensor hereunder), non-transferable (except as provided in Section 11.7) and non- sublicensable license, during the License Term, to reproduce, perform, display, transmit and distribute the Licensed Content on the Licensee Siteand Relat...
Rationale: Exclusivity binds the reviewer but has performance conditions or an exit.
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.

**Low** (`EXC-L2`, KINGPHARMACEUTICALSINC_08_09_2006-EX-10.1-PROMOTION AGREEMENT)
> Depomed agrees to grant and hereby grants to King an exclusive option (exercisable at King's sole discretion by providing written notice of intent at any time, but in no event later than 180 days after the Effective Date) to obtain an exclusive license in the Territory to certain of Depomed's proprietary drug delivery technology in combination with both metformin hydrochloride and any other active pharmaceutical ingr...
Rationale: Exclusivity is short (6 months) and limited in territory.
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.

### IP Ownership Assignment

**High** (`IP-H2`, KIROMICBIOPHARMA,INC_05_11_2020-EX-10.23-CONSULTING AGREEMENT)
> Consultant hereby irrevocably assigns to Company all right, title and interest in and to any information (including, without limitation, business plans and/or business information), technology, know-how, materials, notes, records, designs, ideas, inventions, improvements, devices, developments, discoveries, compositions, trade secrets, processes, methods and/or techniques, whether or not patentable or copyrightable, ...
Rationale: The other party owns all rights with no reservation of the reviewer's existing IP.
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.

**Medium** (`IP-M2`, BICYCLETHERAPEUTICSPLC_03_10_2020-EX-10.11-SERVICE AGREEMENT)
> To the extent that such IPRs do not vest automatically in the Company by operation of law, you hereby assign and agree to assign to the Company all of your right, title and interest in any existing and future IPRs which may subsist in any Works for their full term of protection (including any extensions, revivals and renewals) together with the right to sue and claim remedies for past infringement and all materials e...
Rationale: The other party owns some work product; scope of the transfer is unclear.
Assumption: The clause does not say which party is the reviewer and no reviewer_aliases were given, so it was read against the reviewer.
