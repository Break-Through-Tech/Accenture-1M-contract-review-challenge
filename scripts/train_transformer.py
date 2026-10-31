"""Fine-tune BERT for multi-label clause classification over the 41 CUAD categories.

Chunks and labels are the baseline's (``baseline_tfidf.py``): 512-token,
64-overlap ``bert-base-uncased`` windows, positive for a category if the chunk
overlaps an annotated span.  Each chunk is one example with 41 yes/no outputs.
BERT fits 510 content tokens plus [CLS] and [SEP], so the last 2 tokens of a
window are dropped; they are also the start of the next window's overlap.

Each epoch trains on every train chunk that holds a clause plus a fresh random
quarter of the empty ones (53% of train chunks hold no clause).  The randomly
initialised classification layer learns at a higher rate than the pretrained
encoder.  The model is scored on validation after every epoch, and the epoch
with the best validation macro average precision is kept.  Per-category
thresholds are then chosen on validation.  The test set is not used.

Run from the project folder::

    python scripts/train_transformer.py            # full run, 2-3 hours on an M4
    python scripts/train_transformer.py --smoke    # a few batches, to check the pipeline
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from baseline_tfidf import (PROCESSED_DIR, PROJECT_DIR, best_f1_threshold, contract_labels,
                            label_chunks, load_tokenizer, per_category_metrics, summarise,
                            window_bounds)
from load_dataset import load_annotations, load_contracts

MODEL_NAME = "bert-base-uncased"
MAX_CONTENT_TOKENS = 510
MODEL_DIR = PROJECT_DIR / "models" / "transformer_v1"
RESULTS_DIR = PROJECT_DIR / "results" / "transformer_v1"

EPOCHS = 6
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
HEAD_LEARNING_RATE = 1e-3
EMPTY_CHUNK_FRACTION = 0.25
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42


def all_categories() -> list[str]:
    categories = json.loads((PROCESSED_DIR / "categories.json").read_text(encoding="utf-8"))
    return [c["name"] for c in categories]


def make_examples(contracts: pd.DataFrame, tokenizer) -> tuple[pd.DataFrame, list[list[int]]]:
    """Baseline chunks (one row each) and the token ids BERT sees for each."""
    rows, ids = [], []
    for cid, text, split in zip(contracts["contract_id"], contracts["text"], contracts["split"]):
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        for i, (s, e) in enumerate(window_bounds(len(offsets))):
            rows.append((cid, i, offsets[s][0], offsets[e - 1][1], e - s, split))
            ids.append(enc["input_ids"][s:e][:MAX_CONTENT_TOKENS])
    chunks = pd.DataFrame(rows, columns=["contract_id", "chunk_index", "start", "end",
                                         "n_tokens", "split"])
    return chunks, ids


class Examples(Dataset):
    def __init__(self, items: list[tuple[list[int], np.ndarray]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[list[int], np.ndarray]:
        return self.items[i]


def epoch_sample(train_idx: np.ndarray, has_clause: np.ndarray, fraction: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Every chunk with a clause, plus a random ``fraction`` of the empty ones."""
    empty = train_idx[~has_clause]
    kept = rng.choice(empty, int(round(len(empty) * fraction)), replace=False)
    return np.sort(np.concatenate([train_idx[has_clause], kept]))


def positive_weights(Y: np.ndarray) -> np.ndarray:
    """sqrt(negatives / positives) per category: a softer push than the full ratio."""
    pos = Y.sum(axis=0)
    weights = np.sqrt((len(Y) - pos) / np.maximum(pos, 1))
    return np.where(pos > 0, weights, 1.0)


def collate(batch, cls_id: int, sep_id: int, pad_id: int) -> dict[str, torch.Tensor]:
    """Add [CLS] and [SEP], then pad to the longest chunk in the batch."""
    seqs = [[cls_id, *ids, sep_id] for ids, _ in batch]
    width = max(map(len, seqs))
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, seq in enumerate(seqs):
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attention_mask[i, :len(seq)] = 1
    labels = torch.tensor(np.stack([y for _, y in batch]), dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def macro_ap(Y: np.ndarray, P: np.ndarray) -> float:
    return float(np.mean([average_precision_score(Y[:, j], P[:, j])
                          for j in range(Y.shape[1]) if Y[:, j].sum() > 0]))


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def train(smoke: bool = False) -> None:
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    device = pick_device()
    model_dir = MODEL_DIR.with_name("transformer_v1_smoke") if smoke else MODEL_DIR
    results_dir = model_dir / "results" if smoke else RESULTS_DIR
    epochs = 1 if smoke else EPOCHS

    categories = all_categories()
    contracts = load_contracts().drop(columns="split").merge(
        pd.read_csv(PROCESSED_DIR / "splits.csv"), on="contract_id", how="inner", validate="1:1")
    annotations = load_annotations()
    annotations = annotations.loc[annotations["contract_id"].isin(contracts["contract_id"])]

    tokenizer = load_tokenizer(MODEL_NAME)
    chunks, ids = make_examples(contracts, tokenizer)
    tokenizer.model_max_length = 512  # load_tokenizer lifts it for whole contracts; restore for saving
    Y = label_chunks(chunks, annotations, categories).to_numpy(dtype=np.float32)
    idx = {s: np.flatnonzero(chunks["split"].to_numpy() == s) for s in ("train", "val")}
    if smoke:
        rng = np.random.default_rng(SEED)
        idx = {s: np.sort(rng.choice(v, 64, replace=False)) for s, v in idx.items()}
    print(f"device: {device}  chunks: train {len(idx['train'])}, val {len(idx['val'])}  "
          f"categories: {len(categories)}", flush=True)

    def loader(rows: np.ndarray, shuffle: bool, seed: int = SEED) -> DataLoader:
        data = Examples([(ids[i], Y[i]) for i in rows])
        return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(seed),
                          collate_fn=lambda b: collate(b, tokenizer.cls_token_id,
                                                       tokenizer.sep_token_id,
                                                       tokenizer.pad_token_id))

    val_loader = loader(idx["val"], False)
    Y_train, Y_val = Y[idx["train"]], Y[idx["val"]]
    pos_weight = positive_weights(Y_train)
    has_clause = Y_train.sum(axis=1) > 0
    sampler = np.random.default_rng(SEED)
    per_epoch = len(epoch_sample(idx["train"], has_clause, EMPTY_CHUNK_FRACTION, sampler))
    print(f"train chunks per epoch: {per_epoch} ({has_clause.sum()} with a clause, "
          f"{per_epoch - has_clause.sum()} of {(~has_clause).sum()} empty)", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(categories), problem_type="multi_label_classification",
        id2label=dict(enumerate(categories)), label2id={c: i for i, c in enumerate(categories)},
    ).to(device)
    no_decay = ("bias", "LayerNorm.weight")
    params = []
    for is_head, lr in ((False, LEARNING_RATE), (True, HEAD_LEARNING_RATE)):
        named = [(n, p) for n, p in model.named_parameters() if n.startswith("classifier.") == is_head]
        params += [
            {"params": [p for n, p in named if not n.endswith(no_decay)], "lr": lr,
             "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in named if n.endswith(no_decay)], "lr": lr, "weight_decay": 0.0},
        ]
    optimizer = torch.optim.AdamW(params)
    total_steps = epochs * -(-per_epoch // BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps),
                                                total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32,
                                                                 device=device))

    log, val_ap, best = [], {}, (-1.0, 0, None)
    step = 0
    sampler = np.random.default_rng(SEED)
    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        rows = epoch_sample(idx["train"], has_clause, EMPTY_CHUNK_FRACTION, sampler)
        for batch in loader(rows, True, SEED + epoch):
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            running.append(loss.item())
            if step % 50 == 0 or step == total_steps:
                elapsed = time.time() - t0
                print(f"epoch {epoch} step {step}/{total_steps}  loss {np.mean(running[-50:]):.4f}  "
                      f"{elapsed / 60:.0f} min elapsed, about "
                      f"{elapsed / step * (total_steps - step) / 60:.0f} min left", flush=True)
                log.append({"epoch": epoch, "step": step, "train_loss": np.mean(running[-50:])})

        P_val = predict(model, val_loader, device)
        val_ap[epoch] = macro_ap(Y_val, P_val)
        log.append({"epoch": epoch, "step": step, "train_loss": np.mean(running),
                    "val_macro_avg_precision": val_ap[epoch]})
        print(f"epoch {epoch}: val macro AP {val_ap[epoch]:.4f}", flush=True)
        if best[2] is None or val_ap[epoch] > best[0]:
            best = (val_ap[epoch], epoch, P_val)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    _, best_epoch, P_val = best
    assert P_val is not None  # epoch 1 is always kept
    thresholds = {c: best_f1_threshold(Y_val[:, j], P_val[:, j]) for j, c in enumerate(categories)}
    Yv = pd.DataFrame(Y_val.astype(int), columns=categories)
    Pv = pd.DataFrame(P_val, columns=categories)
    chunk_metrics = per_category_metrics(Yv, Pv, thresholds)

    val_chunks = chunks.iloc[idx["val"]][["contract_id", "chunk_index", "start", "end"]]
    val_chunks = val_chunks.reset_index(drop=True)
    contract_scores = Pv.groupby(val_chunks["contract_id"]).max()
    Yc = contract_labels(annotations, contract_scores.index, categories)
    contract_metrics = per_category_metrics(
        Yc, pd.DataFrame(contract_scores.to_numpy(), columns=categories), thresholds)
    summary = pd.DataFrame({"chunk_level": summarise(chunk_metrics),
                            "contract_level": summarise(contract_metrics)})

    results_dir.mkdir(parents=True, exist_ok=True)
    chunk_metrics.round(4).to_csv(results_dir / "val_metrics_chunk_level.csv")
    contract_metrics.round(4).to_csv(results_dir / "val_metrics_contract_level.csv")
    summary.round(4).to_csv(results_dir / "val_metrics_summary.csv")
    pd.DataFrame(log).round(4).to_csv(results_dir / "training_log.csv", index=False)
    pd.concat([val_chunks, Pv.add_prefix("prob__"), Yv.add_prefix("label__")], axis=1) \
        .round(6).to_csv(results_dir / "val_predictions.csv", index=False)
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n",
                                               encoding="utf-8")
    config = {
        "model": MODEL_NAME,
        "chunks": "baseline chunks: 512 tokens, 64-token overlap; first 510 fed to BERT",
        "chunk_label_rule": "positive if the chunk overlaps any annotated span",
        "categories": categories,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "empty_chunk_fraction_per_epoch": EMPTY_CHUNK_FRACTION,
        "train_chunks_per_epoch": per_epoch,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "loss": "BCEWithLogitsLoss, pos_weight = sqrt(negatives / positives) per category",
        "seed": SEED,
        "device": str(device),
        "train_chunks": len(idx["train"]),
        "val_chunks": len(idx["val"]),
        "val_macro_avg_precision_by_epoch": val_ap,
        "best_epoch": best_epoch,
        "thresholds": "per category, max F1 on validation chunks",
        "test_set": "not used",
        "train_minutes": round((time.time() - t0) / 60, 1),
        "saved_model": str(model_dir.relative_to(PROJECT_DIR)),
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print("\n" + summary.round(3).to_string())
    print(f"\nbest epoch {best_epoch}; model saved to {model_dir.relative_to(PROJECT_DIR)}, "
          f"results to {results_dir.relative_to(PROJECT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--smoke", action="store_true",
                        help="64 train and 64 val chunks, 1 epoch, saved under models/")
    train(parser.parse_args().smoke)


if __name__ == "__main__":
    main()
