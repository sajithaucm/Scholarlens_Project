"""Fine-tune and evaluate scientific NER models (SciBERT vs BERT-base) on SciERC.

Each word gets a BIO tag over six entity types (Task, Method, Metric, Material,
OtherScientificTerm, Generic). Words split into several sub-word pieces are
labelled on their first piece only; the other pieces are ignored (-100) in the
loss and in evaluation, so scores are computed per original word.

Usage:
    python -m src.ner.train --model allenai/scibert_scivocab_uncased
    python -m src.ner.train --model all
    python -m src.ner.train --model bert-base-uncased --max-samples 200 --epochs 1   # quick test
"""

import argparse
import json
import shutil
import time
from typing import Dict, List

import numpy as np
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

from src.data_loading import load_ner
from src.evaluation.metrics import ner_metrics
from src.utils import (load_config, output_dir, resolve, safe_name, save_results, set_seed,
                       training_settings, use_fp16, warmup_kwargs, write_jsonl)

TASK = "ner"


def load_tokenizer(model_name: str):
    """RoBERTa-style tokenizers need add_prefix_space=True for pre-split words."""
    try:
        return AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name)


def tokenize_and_align(batch, tokenizer, max_len: int):
    enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
    all_labels = []
    for i, word_labels in enumerate(batch["ner_tags"]):
        prev, labels = None, []
        for wid in enc.word_ids(batch_index=i):
            if wid is None or wid == prev:
                labels.append(-100)          # special token or continuation piece
            else:
                labels.append(word_labels[wid])
            prev = wid
        all_labels.append(labels)
    enc["labels"] = all_labels
    return enc


def decode_predictions(logits: np.ndarray, label_ids: np.ndarray, id2label: Dict[int, str]):
    """Turn logits + aligned labels back into word-level tag sequences."""
    preds = logits.argmax(-1)
    true_seqs, pred_seqs = [], []
    for p_row, l_row in zip(preds, label_ids):
        t, p = [], []
        for p_id, l_id in zip(p_row, l_row):
            if l_id == -100:
                continue
            t.append(id2label[int(l_id)])
            p.append(id2label[int(p_id)])
        true_seqs.append(t)
        pred_seqs.append(p)
    return true_seqs, pred_seqs


def train(model_name: str, data, cfg: dict) -> Dict:
    set_seed(cfg["seed"])
    s = training_settings(cfg, TASK, model_name)
    out = output_dir(cfg, TASK, model_name)

    label_names: List[str] = data["train"].features["ner_tags"].feature.names
    id2label = dict(enumerate(label_names))
    label2id = {v: k for k, v in id2label.items()}

    tokenizer = load_tokenizer(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_names), id2label=id2label, label2id=label2id)
    tokenized = data.map(lambda b: tokenize_and_align(b, tokenizer, cfg[TASK]["max_seq_len"]),
                         batched=True, remove_columns=data["train"].column_names, desc="Tokenizing")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        t, p = decode_predictions(logits, labels, id2label)
        m = ner_metrics(t, p)
        return {k: v for k, v in m.items() if k != "per_type_f1"}

    args = TrainingArguments(
        output_dir=str(out),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",            # entity F1 on the dev split
        greater_is_better=True,
        save_total_limit=1,
        learning_rate=float(s["learning_rate"]),
        per_device_train_batch_size=int(s["batch_size"]),
        per_device_eval_batch_size=int(s["batch_size"]) * 2,
        gradient_accumulation_steps=int(s.get("grad_accum_steps", 1)),
        num_train_epochs=float(s["epochs"]),
        weight_decay=float(s["weight_decay"]),
        **warmup_kwargs(float(s.get("warmup_ratio", 0.0))),
        fp16=use_fp16(s, model_name),
        logging_steps=int(s.get("logging_steps", 50)),
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized["validation"],
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      processing_class=tokenizer, compute_metrics=compute_metrics)
    t0 = time.time()
    trainer.train()
    train_minutes = round((time.time() - t0) / 60, 1)

    best_dir = out / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    for ckpt in out.glob("checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)

    pred = trainer.predict(tokenized["test"])
    true_seqs, pred_seqs = decode_predictions(pred.predictions, pred.label_ids, id2label)
    metrics = ner_metrics(true_seqs, pred_seqs)
    metrics["train_minutes"] = train_minutes
    metrics["n_test_sentences"] = len(true_seqs)

    results_dir = resolve(cfg["paths"]["results_dir"])
    rows = []
    for ex, t, p in zip(data["test"], true_seqs, pred_seqs):
        n = len(t)   # words beyond max_seq_len were truncated
        rows.append({"tokens": ex["tokens"][:n], "gold": t, "pred": p, "all_correct": t == p})
    write_jsonl(rows[:300], results_dir / f"{TASK}_{safe_name(model_name)}_samples.jsonl")
    with open(results_dir / f"{TASK}_{safe_name(model_name)}_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    save_results(cfg, TASK, model_name, metrics, extra={
        "settings": s, "data_sizes": {k: len(v) for k, v in data.items()},
        "labels": label_names, "checkpoint": str(best_dir),
    })
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=float, default=None)
    args = p.parse_args()

    cfg = load_config()
    if args.epochs is not None:
        cfg[TASK].setdefault("training", {})["epochs"] = args.epochs
    data = load_ner(cfg, args.max_samples)
    for name in (cfg[TASK]["models"] if args.model == "all" else [args.model]):
        print(f"\n===== Training {name} =====")
        train(name, data, cfg)


if __name__ == "__main__":
    main()
