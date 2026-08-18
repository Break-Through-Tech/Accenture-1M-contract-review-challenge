# Contract Review Challenge

**Company / Org:** Accenture  
**Challenge Advisor:** Hetal Jetani, hetal.jetani@accenture.com / hetal0826@gmail.com  
**AI Studio Coach:** Parth Dali, parth.dali@breakthroughtech.org  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Accenture
Accenture is a leading global professional services company that provides a broad range of services and solutions in strategy, consulting, technology, and operations. 

---

## 🎯 The Challenge
### Project Summary
In this project, you will use real-world commercial contracts from the CUAD dataset (510 contracts, 41 expert-annotated clause categories) and NLP techniques including chunk-based multi-label classification with fine-tuned transformer encoders, paired with an explainable rule-based risk-scoring layer, to build a pipeline that automatically detects key clauses, flags them as Low/Medium/High risk, and rolls these up into a contract-level triage score. This will help our company address the bottleneck legal and procurement teams face when manually reviewing tens of thousands of contracts a year to find the small number of clauses that carry meaningful risk, enabling reviewers to prioritize which contracts to open first.

### Success Criteria
Success has two tracks:
- For clause detection: per-category precision/recall/F1 clearly beating the baseline (accuracy is misleading under CUAD's imbalance), with error analysis on where the model struggles.   
- For risk scoring: since there are no ground-truth labels, success means strong Spearman correlation and bucket agreement between the model's risk rankings and the advisor's hand-ranked clauses, plus a sensitivity analysis showing the High/Medium boundary is stable.

Overall, a successful December outcome is a working end-to-end pipeline producing risk-scored clause registers the advisor finds plausible and useful, a clean documented repo, and a final report covering results, limitations, and estimated reviewer time saved — an auditable triage tool the advisor would actually trust, not a black box.

### Stretch Goals
Stretch goals include span extraction, a trained risk model benchmarked against the rule-based baseline, LLM-generated clause explanations, broader category coverage, a Streamlit/Gradio demo, and an active-learning loop using advisor/model disagreements. These extend modeling or usability without affecting core deliverables.

### Project Milestones

Use these milestones to guide the project. The team should create a GitHub Projects board to break each milestone into smaller weekly tasks and track progress throughout the project.

| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | **Data Understanding & Baseline** | Understand and clean the CUAD data, create contract-level train/validation/test splits, perform EDA on clause frequency and class imbalance, develop a chunking strategy, and establish a simple TF-IDF/keyword baseline with basic precision, recall, and F1 metrics. |
| **October** | **Transformer Classification** | Fine-tune a transformer encoder for multi-label clause classification, experiment with approaches to class imbalance, evaluate overall and per-category precision/recall/F1, and conduct initial error analysis. |
| **November** | **Risk Scoring & Pipeline Integration** | Build and test the explainable rule-based risk-scoring layer, assign Low/Medium/High risk to detected clauses, develop the contract-level triage score, integrate clause detection and risk scoring into an end-to-end pipeline, and validate the approach using advisor-reviewed examples. |
| **December** | **Final Validation & Demo** | Conduct final evaluation, refine the risk-scoring rules, review false positives and false negatives, document limitations, finalize the end-to-end contract risk report, prepare the final demonstration, and organize the GitHub repository for external review. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Create columns or views for each project month. Use GitHub Issues to track individual tasks, assign owners, and document progress.

---

## 📊 Dataset
**Name and Source:** CUAD Dataset (Contract Understanding Atticus Dataset)  
**Format:** JSON, Raw Text/PDF  
**Size:** under 1gb  
**Location:** https://github.com/TheAtticusProject/cuad 

| Dataset / Source | Purpose in Project | Format | Access |
|---|---|---|---|
| **CUAD Master Clauses Dataset** | Primary training dataset for identifying the 41 legal clause categories. Contains contract text and expert-annotated answers for each category. Recommended starting point for students. | CSV | [CUAD on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad) |
| **CUAD SQuAD-style Dataset** | Used for clause extraction and identifying where relevant clauses appear within a contract. | JSON | [CUAD JSON Data](https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1) |
| **CUAD Full Contracts – Text** | Provides complete contract text for preprocessing, chunking, testing, and running the final pipeline against full contracts. | 510 TXT files | [Full Contract Text Files](https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1/full_contract_txt) |
| **CUAD Category Descriptions** | Defines the 41 clause categories and provides guidance on what each category represents. Useful for building the label mapping and understanding the classification task. | CSV | [CUAD GitHub Repository](https://github.com/TheAtticusProject/cuad) |
| **CUAD Dataset – Hugging Face** | Provides a machine-learning-friendly way to load CUAD directly into Python and Hugging Face workflows. Useful for students working with Transformers and PyTorch. | Hugging Face Dataset | [CUAD on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad) |

### Working Dataset Expectations

* **Primary Dataset:** Use the CUAD (Contract Understanding Atticus Dataset), containing 510 commercial contracts and 41 expert-annotated clause categories.
* **Initial Scope:** Start with approximately 50–100 contracts for data exploration, preprocessing, and baseline development before expanding to the full dataset.
* **Data Exploration:** Analyze contract length, clause frequency, category distribution, and potential class imbalance.
* **Preprocessing:** Clean and standardize contract text while preserving relevant clause boundaries and annotations.
* **Chunking:** Develop a chunking strategy that allows long contracts to be processed by transformer models while retaining sufficient context.
* **Data Splits:** Create contract-level training, validation, and test sets to prevent data leakage.
* **Classification Labels:** Use CUAD’s 41 clause categories as the initial multi-label classification targets.
* **Evidence Retention:** Preserve the relevant text span for each detected clause so predictions can be explained and reviewed.
* **Risk Scoring:** Develop a separate, explainable rule-based layer to assign **Low/Medium/High** risk, since CUAD does not provide risk labels.
* **Contract-Level Triage:** Aggregate clause-level risk scores into an overall contract risk/triage score.
* **Reproducibility:** Document the dataset version, preprocessing, data splits, assumptions, and methodology in GitHub.
* **Final Output:** The pipeline should produce **clause category → evidence → risk level → rationale → contract-level triage score**.

### Known Preprocessing and Data Risks
* Normalize contract text consistently, including formatting, whitespace, headers, and page breaks while preserving meaningful legal language.
* Standardize contract IDs, clause labels, annotation spans, and document metadata across all source files.
* Handle long contracts carefully: chunk text without separating important clause context or splitting relevant annotations incorrectly.
* Expect domain and annotation variability: CUAD contracts and expert annotations may differ in structure and language, which can affect model performance on new or unseen contracts.


### Key Details
- Real-world commercial contracts from the CUAD dataset (510 contracts, 41 expert-annotated clause categories), raw text/PDF available.
- Team must implement strict preprocessing rules to handle document length variance and ensure text cleaning captures the necessary legal terminology for high-accuracy classification.

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP / Multi-label Classification / Information Extraction / Explainable Risk Scoring

**Recommended Libraries:** Hugging Face Transformers, PyTorch, scikit-learn, pandas, NumPy, Hugging Face Datasets

**Suggested Pipeline:** Contract Text → Preprocessing & Chunking → Multi-label Clause Classification → Evidence Extraction → Rule-based Risk Scoring → Contract-level Triage Score

**Evaluation Metrics:** Precision, Recall, and F1-Score for clause classification; High-Risk Recall and Precision@K for contract triage; basic error analysis for risk-scoring results.

**Development Environment:** Google Colab for model training and experiments; VS Code and Jupyter Notebooks for development and analysis.

---
## 📚 Resources to Get Started

These resources will help your team understand the CUAD dataset, legal clause classification, transformer models, and the overall project approach.

**Background Reading:**
- [CUAD – Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad/) — Dataset overview and legal contract review problem.
- [CUAD Labeling Handbook](https://www.atticusprojectai.org/labeling-handbook/) — Definitions and examples of the 41 clause categories.
- [CUAD Research Paper](https://arxiv.org/abs/2103.06268) — Technical background on the CUAD dataset.

**Technical Tutorials:**
- [Hugging Face – Text Classification](https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification) — Fine-tuning transformer models.
- [Hugging Face – Datasets](https://huggingface.co/docs/datasets/) — Loading and processing datasets.

**Code & Data:**
- [Official CUAD GitHub Repository](https://github.com/TheAtticusProject/cuad) — Reference code and dataset resources.
- [CUAD on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad) — Machine-learning-friendly access to CUAD.

**ML Problem Type:** NLP / Multi-label Classification / Information Extraction / Explainable Risk Scoring

**Recommended Tools:**
- **Python:** pandas, NumPy, scikit-learn
- **ML/NLP:** Hugging Face Transformers, Hugging Face Datasets, PyTorch
- **Development:** Google Colab, VS Code, Jupyter Notebooks
- **Version Control:** Git and GitHub

> **Tip:** Start with the CUAD dataset and Labeling Handbook before selecting a model. You are encouraged to explore additional tools, techniques, and resources as you develop the project and share useful findings with the team.

**Suggested Pipeline:**

CUAD Contracts → Preprocessing → Chunking → Multi-label Clause Classification → Evidence Extraction → Rule-based Risk Scoring → Contract-level Triage Score → Explainable Risk Report

## Recommended Modeling Approach

- **Establish a Baseline:** Build a simple TF-IDF/keyword-based classifier before using transformer models.
- **Train the Model:** Fine-tune a transformer encoder for multi-label classification across the 41 CUAD clause categories.
- **Preserve Evidence:** Retain the relevant contract text/span for each detected clause to support explainability.
- **Add Risk Scoring:** Develop an explainable rule-based layer that assigns **Low / Medium / High** risk based on detected clause characteristics.
- **Calculate Triage Score:** Aggregate clause-level risks into an overall **contract-level triage score** to help prioritize contracts for review.
- **Evaluate Performance:** Measure clause classification using **Precision, Recall, and F1 Score**, with emphasis on identifying high-risk clauses.
- **Perform Error Analysis:** Review false positives, false negatives, rare clause categories, and difficult contract language to identify opportunities for improvement.
---

## Evaluation Metrics

| **Component** | **Metric** | **Purpose** |
|---|---|---|
| Clause Classification | **Precision** | Of the clauses the model identifies, how many are correct? |
| Clause Classification | **Recall** | Of the relevant clauses in the contract, how many does the model find? |
| Clause Classification | **F1 Score** | Combines Precision and Recall into one overall classification score. |
| Risk Scoring | **Accuracy** | How often does the system correctly assign Low, Medium, or High risk? |
| Contract Triage | **High-Risk Recall** | Of the contracts that should receive priority review, how many does the system successfully flag? |

### Primary Evaluation

Team should focus primarily on **Precision, Recall, and F1 Score** when evaluating clause classification.

For the final risk-triage pipeline, **High-Risk Recall** is especially important because the goal is to avoid missing contracts that may require additional legal or procurement review.

Team should also perform a simple **error analysis** by reviewing examples of:
- Incorrectly identified clauses
- Missed clauses
- Incorrect risk assignments
- Difficult or ambiguous contract language

The goal is not only to report a score, but to understand **where the model works well, where it fails, and why**.

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [Discord Channel](https://discord.gg/7XC4pB5deF)
* hetal.jetani@accenture.com, hetal0826@gmail.com (Google Meet)
* [Mobile connect, WhatsApp]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

## Recommended Tools

- **Coding:** Google Colab, VS Code, Jupyter Notebooks
- **ML / NLP:** Hugging Face Transformers, Hugging Face Datasets, PyTorch
- **Data Analysis:** pandas, NumPy, scikit-learn
- **Collaboration:** Git, GitHub Projects, Notion
- **Environment Management:** `uv` or Python virtual environments
- **Documentation:** GitHub README and project documentation
- **Virtual Meetings:** Zoom, Google Meet
- **Team Communication:** WhatsApp for quick coordination; use GitHub Issues for project decisions and technical discussions

## What I Expect From the Team

- **Keep work visible:** Track tasks, questions, and progress using GitHub Issues and the GitHub Projects board.
- **Document decisions:** Record important modeling decisions, assumptions, and failed experiments—not only successful results.
- **Move code into modules:** Use notebooks for exploration and experimentation, but move reusable and production-ready code into Python modules as the project matures.
- **Maintain reproducibility:** Keep the repository organized with clear setup instructions, dependencies, data documentation, and reproducible workflows so that an external reviewer can run the demo.
- **Use meaningful commits:** Write clear commit messages that describe what was changed and why.
- **Keep documentation current:** Update the README and relevant documentation as the project architecture, models, and results evolve.

## 🚀 Getting Started

Read this overview and list your open questions before our first team meeting.

1. **Review this overview document** and note any questions for our first meeting: Understand the project goals, technical approach, dataset expectations, milestones.

2. **Review the CUAD Dataset:** Explore the 41 clause categories and identify the initial subset of contracts you will use for data exploration and baseline development.

3. **Define the Data Structure:** Agree on the standard format for contracts, clause labels, text chunks, evidence spans, model predictions, and risk scores.

4. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)   
Set up the project board and create initial issues for Eg:
   - CUAD data access and setup
   - Data exploration and preprocessing
   - Chunking strategy
   - Baseline model
   - Evaluation metrics
   - Transformer model
   - Risk-scoring rules

5. **Prepare Open Questions:** Record questions, assumptions, and areas where you need clarification before the first team meeting.

6. **Document Your Decisions:** Keep important technical decisions and findings in GitHub Issues or project documentation so the entire team can follow the project's progress.

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
