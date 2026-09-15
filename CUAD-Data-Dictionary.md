# CUAD Data Dictionary & Sample Annotated Clauses

**Issue:** #8 — CUAD Familiarity

## 1. File-Level Structure
The JSON file uses two top-level keys:
-   'version': which denotes the specific dataset version
-   'data': a collection of 408 contracts
Refer to cell 2 

## 2. Contract-Level Structure
In a single contract, there exists two keys
- "title": serves as the contract's unique identifer (example: `LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT' )
- "paragraphs": A list containing the contract's text and annotations. CUAD stores the entire contract as a single block of text, meaning the length of the list is exactly 1. This is important as we think about chunking moving forward. 

for examples, refer to cells 4 & 5 in the cuad_familiarity.ipynb

## 3. Paragraph-Level Structure
Paragraphs also hold two keys:
-   Context: the raw text in the contract 
-   qas: a list of annoation entires in the form of question and answers

for findings, refer to cells 6, 7, 8 in cuad_familiarity.ipynb

## 4. Qa-Entry Structure
QA entries have 5 total keys per entry
-   questions: Fixed sentence template, with the exact catergory of the question being embedded in quotes, meaning that we need to extract is by splitting the string 
-   answers: list of dicts, with text deriving from the extracted clause
-   answer_start: character offsett in context
-   is_impossible: Uses a True/False structure. If true, then the clause is not real and answers will be an empty list
-   id: combination of contract title + category + index number

For findings, refer to cell 8 in cuad_familiarity.ipynb


## 5. Category Reference (category_descriptions.csv)
In categery descriptions, there are 4 columns: category, Description, Answer format & group

While exploring this specificlly, it became apparent that every value has has the column name included in as text, which needs to be cleaned up during data prep. 

## 6. Sample Annotated Clauses
Outputs from QAs loop 
Example 1

Category: Document Name
Contract: LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
Answer: "DISTRIBUTOR AGREEMENT"

Example 2

Category: Parties
Contract: LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
Answers (multiple — 5 total): "Distributor", "Electric City of Illinois L.L.C.", "Electric City of Illinois LLC", "Company", "Electric City Corp."
Note: illustrates that a single category can have more than one answer span in the same contract

Example 3

Category: Agreement Date
Contract: LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
Answer: "7th day of September, 1999."

Example 4

Category: Governing Law
Contract: LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
Answer: "This Agreement is to be construed according to the laws of the State of Illinois."

Example 5

Category: Notice Period To Terminate Renewal
Contract: LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
Answer: None
Note: illustrates a category with no matching clause in this contract (is_impossible: True, empty answers list)

## 7. Notes / Open Questions
Some categories have multiple valid answer spans per contract (For example,because there could be multiple parties involved in a contract, the number of total parties will vary from contract to contract.)
-   need to decide how this affects classification labels later

Some answers are full sentences, not clean values (Effective Date, Agreement Date)
-   may need post-processing

