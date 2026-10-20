# October deliverable: Multi-Label & Imbalance

**Owner:** Aldo Cortes

**Planned window:** October 10–13, 2026

**Acceptance criterion in the October task plan:** Working multi-label training run.

## What this implementation does

`scripts/transformer_multilabel.py` trains a pretrained BERT encoder with one
output for each of the same 14 shortlisted clause categories as the baseline.
Each chunk appears once and has a 14-element binary target vector. Two or more
labels can be positive for the same text.

The training loss is `BCEWithLogitsLoss`. Predictions use independent sigmoid
probabilities. Sigmoid happens during inference, rather than before the loss,
because the loss already combines sigmoid and binary cross-entropy.

For each category, the class weight is the number of negative training chunks
divided by positive training chunks. We clip the ratio to a configurable range
of 1–50 to limit extreme gradients. A category with no training positives uses
weight 1, and the run records it as absent. This weighting rule is a starting
choice, not a validated optimum. `--class-weight none` allows an unweighted run
with the same seed and data for a later controlled comparison.

## Data and evaluation decisions

- Use `data/processed/splits.csv`: 323 training, 84 validation and 102 test
  contracts for full runs. The excluded duplicate stays outside the model.
- Create chunks only for training and validation contracts. Never evaluate
  or tune on the test split in this deliverable.
- Use fast-tokenizer offsets to match annotation spans to the original cleaned
  text, rather than estimate character positions from decoded tokens.
- Include overlapping windows with 64 content tokens of overlap. The maximum
  model input is 512 tokens, so BERT windows contain up to 510 content tokens
  plus two special tokens. This avoids dropping evidence at a window boundary.
- Retain chunks with no positive labels. They provide necessary negative examples.
- Save the category order in the model configuration and run manifest.
- Use threshold 0.5 for this setup check. Validation threshold experiments
  remain Sai's October 14–16 task.

The content window differs by two tokens from the baseline's 512 content-token
windows. A later model comparison should document this difference or align
the baseline to the same windows.

## Running it

From the repository root, with the project environment activated:

```bash
python -m pip install -r requirements.txt
python scripts/transformer_multilabel.py --smoke-test
```

The first run downloads BERT into the ignored `.model-cache/` directory.
The smoke test selects 16 training and 4 validation contracts with seed 42,
runs five training batches, and evaluates all chunks from those four validation
contracts. It saves evidence under `results/transformer_multilabel/weighted_smoke/`.
Existing result directories cannot be overwritten. For a repeat:

```bash
python scripts/transformer_multilabel.py --smoke-test --output-dir results/transformer_multilabel/smoke_repeat
```

For full training, preferably on a GPU:

```bash
python scripts/transformer_multilabel.py --epochs 3 --batch-size 4 --output-dir results/transformer_multilabel/weighted_full
```

For an unweighted comparison, use the same options and seed, with
`--class-weight none` and a new output directory. A weighted loss and an
unweighted loss on the same logits do not establish which trained model is
better. Compare validation precision, recall and F1 after matched training runs.

## Evidence to attach to the task

| File | Purpose |
| --- | --- |
| `training_log.csv` | Completed optimizer steps, finite loss, gradient norm and positive label counts |
| `class_balance.csv` | Training support, positive rates and actual per-category loss weights |
| `config.json` | Seed, model, categories, split fingerprint, selected contracts and library versions |
| `validation_metrics.csv` | Per-category validation precision, recall and F1 at threshold 0.5 |
| `validation_summary.json` | Overall validation metrics and evaluated chunk count |
| `validation_predictions.csv` | Independent probabilities and labels with contract IDs and span offsets |
| `model/` | Saved model and tokenizer for local reload, excluded from Git because of size |

`notebooks/multilabel_imbalance.ipynb` provides a readable walkthrough and loads
the saved evidence. The small run establishes that the training path works.
It does **not** establish final model quality, improvement over the baseline,
or completion of the full Transformer v1 milestone.

## Technical references

- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [Hugging Face BERT sequence classification](https://huggingface.co/docs/transformers/model_doc/bert)

## Verification

`tests/test_transformer_multilabel.py` checks simultaneous labels, float targets,
token-window coverage, class-weight calculations, split separation and weighted
loss gradients. Run the repository tests with `python -m pytest -q`.

## Completed run evidence

The saved weighted smoke test completed **5 optimizer steps** on
CPU, using 16 training contracts (372 chunks). All 14 categories
had positive examples in the training subset. Per-category positive weights
range from 11.83 to 50. The checkpoint and tokenizer save successfully.

Validation covers all 69 chunks from four separate contracts at
threshold 0.5. Micro precision is 0.0429, recall is
0.4091, and F1 is 0.0776. These scores are
diagnostics after five batches and do not show model readiness or improvement.

All **43 repository tests pass**, including five focused multi-label tests.
