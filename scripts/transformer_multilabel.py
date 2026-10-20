"""Train a CUAD transformer with independent clause labels and weighted BCE.

    python scripts/transformer_multilabel.py --smoke-test
    python scripts/transformer_multilabel.py --epochs 3 --batch-size 4

The test split is never used for training or evaluation. Threshold experiments
belong to the next task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline_tfidf import (label_chunks, per_category_metrics,
                            shortlisted_categories)
from load_dataset import PROCESSED_DIR, load_annotations, load_contracts

PROJECT_DIR = PROCESSED_DIR.parents[1]


def training_partitions(contracts: pd.DataFrame, manifest: pd.DataFrame):
    """Use the modeling manifest, not the original CUAD train/test labels."""
    if manifest["contract_id"].duplicated().any():
        raise ValueError("The split manifest assigns a contract more than once.")
    if not manifest["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Unknown split in manifest.")
    usable = contracts[contracts["is_duplicate_of"].isna()]
    missing = set(manifest["contract_id"]) - set(usable["contract_id"])
    if missing:
        raise ValueError("Manifest contains missing or duplicate contracts.")
    # Filter test IDs before accessing their text or creating any chunks.
    joined = usable.drop(columns="split").merge(
        manifest[manifest["split"].isin(["train", "val"])],
        on="contract_id", validate="1:1")
    parts = {name: joined[joined["split"] == name].reset_index(drop=True)
             for name in ("train", "val")}
    if any(frame.empty for frame in parts.values()):
        raise ValueError("Both training and validation contracts are required.")
    return parts


def make_model_chunks(contracts, tokenizer, max_length=512, overlap=64):
    """Retain exact token IDs and offsets. Reserve room for special tokens."""
    content_size = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= overlap < content_size:
        raise ValueError("Overlap must be smaller than the content window.")
    rows = []
    for cid, text in zip(contracts["contract_id"], contracts["text"]):
        encoded = tokenizer(text, max_length=max_length, stride=overlap,
                            return_overflowing_tokens=True,
                            return_offsets_mapping=True, truncation=True)
        for index, offsets in enumerate(encoded["offset_mapping"]):
            content_offsets = [(start, end) for start, end in offsets if end > start]
            if not content_offsets:
                continue
            ids = encoded["input_ids"][index]
            char_start = content_offsets[0][0]
            char_end = content_offsets[-1][1]
            rows.append({"contract_id": cid, "chunk_index": index,
                         "start": char_start, "end": char_end,
                         "input_ids": ids,
                         "attention_mask": encoded["attention_mask"][index]})
    if not rows:
        raise ValueError("No nonempty token windows in the selected contracts.")
    return pd.DataFrame(rows)


def positive_weights(labels, cap=50.0):
    """Training-only negative/positive ratio, capped to limit extreme gradients.

    A category absent from the training subset uses neutral weight 1.
    The caller records absent labels rather than claiming they were learned.
    """
    y = np.asarray(labels, dtype=np.float32)
    if y.ndim != 2 or y.shape[0] == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected a nonempty binary chunk-by-category matrix.")
    if cap < 1 or not np.isfinite(cap):
        raise ValueError("Weight cap must be finite and at least 1.")
    positive = y.sum(axis=0)
    negative = len(y) - positive
    weights = np.ones(y.shape[1], dtype=np.float32)
    np.divide(negative, positive, out=weights, where=positive > 0)
    return np.clip(weights, 1, cap)


class ChunkDataset(Dataset):
    def __init__(self, chunks, labels):
        self.chunks = chunks.reset_index(drop=True)
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        row = self.chunks.iloc[index]
        return {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"],
                "labels": self.labels[index]}


def collate_chunks(tokenizer, examples):
    labels = torch.stack([row["labels"] for row in examples])
    batch = tokenizer.pad([{k: v for k, v in row.items() if k != "labels"}
                           for row in examples], padding=True, return_tensors="pt")
    batch["labels"] = labels
    return batch


def select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.epochs < 1 or args.batch_size < 1 or args.max_steps < 0 or args.max_val_batches < 0:
        raise ValueError("Epochs and batch size must be positive, limits nonnegative.")
    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    out = Path(args.output_dir or PROJECT_DIR / "results/transformer_multilabel" /
               ("weighted_smoke" if args.smoke_test else "train"))
    args.output_dir = str(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Choose an empty output directory to preserve earlier runs: {out}")
    categories = shortlisted_categories()
    manifest = pd.read_csv(PROCESSED_DIR / "splits.csv")
    partitions = training_partitions(load_contracts(), manifest)
    if args.smoke_test:
        partitions = {name: df.sample(n=min(len(df), count), random_state=args.seed)
                      .sort_values("contract_id").reset_index(drop=True)
                      for (name, df), count in zip(partitions.items(), [16, 4])}
        args.max_steps = args.max_steps or 5
    annotations = load_annotations()
    annotations = annotations[annotations["contract_id"].isin(
        pd.concat(partitions.values())["contract_id"])]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                              cache_dir=args.cache_dir)
    tokenizer.model_max_length = 10**9
    data = {name: make_model_chunks(df, tokenizer, args.max_length, args.overlap)
            for name, df in partitions.items()}
    labels = {name: label_chunks(chunks, annotations, categories)
              for name, chunks in data.items()}
    weights = positive_weights(labels["train"], args.weight_cap)
    if args.class_weight == "none":
        weights = np.ones(len(categories), dtype=np.float32)
    device = select_device(args.device)
    print(f"Device: {device}. Labels: {len(categories)}. "
          f"Chunks: train={len(data['train'])}, val={len(data['val'])}.", flush=True)
    absent = [c for c in categories if labels["train"][c].sum() == 0]
    if absent:
        print(f"No training positives in this subset: {absent}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(categories), problem_type="multi_label_classification",
        id2label=dict(enumerate(categories)), label2id={c: i for i, c in enumerate(categories)},
        cache_dir=args.cache_dir).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(ChunkDataset(chunks, labels[name]),
                               batch_size=args.batch_size, shuffle=name == "train",
                               generator=generator if name == "train" else None,
                               collate_fn=lambda rows: collate_chunks(tokenizer, rows))
               for name, chunks in data.items()}
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        for batch in loaders["train"]:
            targets = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            if logits.shape != targets.shape:
                raise ValueError("Logits and labels must both have shape [batch, categories].")
            loss = loss_fn(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a nonfinite loss.")
            unweighted = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.detach(), targets).item()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                  error_if_nonfinite=True)
            optimizer.step()
            step = len(history) + 1
            history.append({"step": step, "epoch": epoch + 1, "loss": loss.item(),
                            "unweighted_bce_same_batch": unweighted,
                            "gradient_norm_before_clip": float(norm),
                            "positive_labels": int(targets.sum().item()),
                            "batch_size": len(targets)})
            print(f"Step {step}: BCE loss={loss.item():.4f}, "
                  f"positive labels={history[-1]['positive_labels']}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and len(history) >= args.max_steps:
            break
    model.eval()
    probabilities, truths = [], []
    with torch.no_grad():
        for index, batch in enumerate(loaders["val"]):
            if args.max_val_batches and index >= args.max_val_batches:
                break
            truths.append(batch.pop("labels").numpy())
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    Y = pd.DataFrame(np.concatenate(truths), columns=categories)
    P = pd.DataFrame(np.concatenate(probabilities), columns=categories)
    metrics = per_category_metrics(Y, P, {c: args.threshold for c in categories})
    # Report micro over every decision and macro over all labels, including zeros.
    tp, fp, fn = (int(metrics[k].sum()) for k in ("tp", "fp", "fn"))
    summary = {"micro_precision": tp / max(1, tp + fp),
               "micro_recall": tp / max(1, tp + fn),
               "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
               "macro_f1_all_labels": float(metrics["f1"].mean()),
               "validation_chunks_evaluated": len(Y), "threshold": args.threshold}
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    pd.DataFrame(history).to_csv(out / "training_log.csv", index=False)
    balance = pd.DataFrame({"category": categories,
                            "train_positive_chunks": labels["train"].sum().values,
                            "train_negative_chunks": (len(labels["train"]) - labels["train"].sum()).values,
                            "positive_rate": labels["train"].mean().values,
                            "pos_weight": weights})
    balance.to_csv(out / "class_balance.csv", index=False)
    metrics.to_csv(out / "validation_metrics.csv")
    predictions = data["val"].iloc[:len(Y)][["contract_id", "chunk_index", "start", "end"]]
    pd.concat([predictions.reset_index(drop=True), Y.add_prefix("label__"),
               P.add_prefix("prob__")], axis=1).to_csv(out / "validation_predictions.csv", index=False)
    config = {**vars(args), "categories": categories, "device_used": str(device),
              "loss": "BCEWithLogitsLoss", "prediction_activation": "sigmoid",
              "content_window_tokens": args.max_length - tokenizer.num_special_tokens_to_add(False),
              "class_weight_rule": "training negatives / positives, clipped to [1, cap]; absent = 1",
              "absent_training_categories": absent,
              "split_manifest_sha256": hashlib.sha256((PROCESSED_DIR / "splits.csv").read_bytes()).hexdigest(),
              "contract_ids": {name: df["contract_id"].tolist() for name, df in partitions.items()},
              "chunk_counts": {name: len(df) for name, df in data.items()},
              "training_steps": len(history), "elapsed_seconds": round(time.monotonic() - started, 2),
              "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
              "interpretation": "Pipeline smoke test only" if args.smoke_test else "Training run; validation only",
              "test_evaluated": False}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved model and run evidence to {out}. {config['interpretation']}.", flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    p.add_argument("--weight-cap", type=float, default=50)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--cache-dir", default=str(PROJECT_DIR / ".model-cache"))
    p.add_argument("--output-dir", default=None)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
