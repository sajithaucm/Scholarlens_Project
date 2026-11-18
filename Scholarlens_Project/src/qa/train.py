"""Fine-tune and evaluate extractive QA models (DistilBERT vs RoBERTa) on SQuAD 2.0.

The model predicts the start and end token of the answer span inside the
context, or points at the [CLS] token when the question is unanswerable.

Honest evaluation protocol:
  1. Train on `train` (subsample of SQuAD train).
  2. Pick the best checkpoint by loss on `tune` (held out from SQuAD train).
  3. Pick the no-answer threshold that maximizes F1 on `tune`.
  4. Report EM / F1 on `test` (the official SQuAD 2.0 dev set) exactly once.

Usage:
    python -m src.qa.train --model roberta-base
    python -m src.qa.train --model all
    python -m src.qa.train --model distilbert-base-uncased --max-samples 300 --epochs 1   # quick test
"""

import argparse
import collections
import json
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

from src.data_loading import load_qa
from src.evaluation.metrics import qa_metrics
from src.utils import (load_config, output_dir, resolve, safe_name, save_results, set_seed,
                       training_settings, use_fp16, warmup_kwargs, write_jsonl)

TASK = "qa"


# ------------------------------------------------------------------ Features
def _tokenize(tokenizer, examples, cfg):
    c = cfg[TASK]
    pad_right = tokenizer.padding_side == "right"
    return tokenizer(
        examples["question" if pad_right else "context"],
        examples["context" if pad_right else "question"],
        truncation="only_second" if pad_right else "only_first",
        max_length=c["max_seq_len"],
        stride=c["doc_stride"],
        return_overflowing_tokens=True,   # long contexts become several overlapping windows
        return_offsets_mapping=True,
        padding="max_length",
    )


def prepare_train_features(examples, tokenizer, cfg):
    """Tokenize and label each window with start/end token positions (CLS=0 if no answer)."""
    enc = _tokenize(tokenizer, examples, cfg)
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets_all = enc.pop("offset_mapping")
    context_id = 1 if tokenizer.padding_side == "right" else 0
    starts, ends = [], []
    for i, offsets in enumerate(offsets_all):
        input_ids = enc["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0
        seq_ids = enc.sequence_ids(i)
        answers = examples["answers"][sample_map[i]]
        if len(answers["answer_start"]) == 0:
            starts.append(cls_index)
            ends.append(cls_index)
            continue
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        ctx_start = seq_ids.index(context_id)
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(context_id)
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            starts.append(cls_index)   # answer is not inside this window
            ends.append(cls_index)
            continue
        s = ctx_start
        while s <= ctx_end and offsets[s][0] <= start_char:
            s += 1
        e = ctx_end
        while e >= ctx_start and offsets[e][1] >= end_char:
            e -= 1
        starts.append(s - 1)
        ends.append(e + 1)
    enc["start_positions"] = starts
    enc["end_positions"] = ends
    return enc


def prepare_eval_features(examples, tokenizer, cfg):
    """Tokenize for prediction; keep offsets (context tokens only) and the example id."""
    enc = _tokenize(tokenizer, examples, cfg)
    sample_map = enc.pop("overflow_to_sample_mapping")
    context_id = 1 if tokenizer.padding_side == "right" else 0
    enc["example_id"] = []
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        enc["example_id"].append(examples["id"][sample_map[i]])
        enc["offset_mapping"][i] = [o if seq_ids[k] == context_id else None
                                    for k, o in enumerate(enc["offset_mapping"][i])]
    return enc


# ------------------------------------------------------------- Post-processing
def best_spans(examples, features, start_logits, end_logits, cfg) -> Dict[str, Tuple[str, float]]:
    """For each question return (best non-null answer text, score_diff = best_span - null)."""
    c = cfg[TASK]
    n_best, max_len = c.get("n_best", 20), c.get("max_answer_tokens", 30)
    feats_per_example = collections.defaultdict(list)
    for i, ex_id in enumerate(features["example_id"]):
        feats_per_example[ex_id].append(i)
    all_offsets = features["offset_mapping"]   # read once: row-by-row access is slow

    out = {}
    for ex in examples:
        null_score, best_text, best_score = None, "", -1e9
        for fi in feats_per_example[ex["id"]]:
            s_log, e_log = start_logits[fi], end_logits[fi]
            offsets = all_offsets[fi]
            fnull = s_log[0] + e_log[0]
            null_score = fnull if null_score is None else min(null_score, fnull)
            s_idx = np.argsort(s_log)[-1:-n_best - 1:-1]
            e_idx = np.argsort(e_log)[-1:-n_best - 1:-1]
            for s in s_idx:
                for e in e_idx:
                    if (s >= len(offsets) or e >= len(offsets) or offsets[s] is None
                            or offsets[e] is None or e < s or e - s + 1 > max_len):
                        continue
                    score = s_log[s] + e_log[e]
                    if score > best_score:
                        best_score = score
                        best_text = ex["context"][offsets[s][0]:offsets[e][1]]
        out[ex["id"]] = (best_text, float(best_score - (null_score if null_score is not None else 0.0)))
    return out


def apply_threshold(spans: Dict[str, Tuple[str, float]], threshold: float) -> Dict[str, str]:
    """Answer only when the best span beats the no-answer score by more than `threshold`."""
    return {qid: (text if diff > threshold else "") for qid, (text, diff) in spans.items()}


def references(examples) -> Dict[str, List[str]]:
    return {ex["id"]: list(ex["answers"]["text"]) for ex in examples}


def tune_threshold(spans, refs) -> Tuple[float, float]:
    """Grid-search the no-answer threshold on the tune split (maximize F1)."""
    diffs = np.array([d for _, d in spans.values()])
    grid = np.unique(np.concatenate([[0.0], np.quantile(diffs, np.linspace(0, 1, 101))]))
    best_t, best_f1 = 0.0, -1.0
    for t in grid:
        f1 = qa_metrics(apply_threshold(spans, float(t)), refs)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def predict_spans(trainer, tokenizer, examples, cfg):
    feats = examples.map(lambda b: prepare_eval_features(b, tokenizer, cfg), batched=True,
                         remove_columns=examples.column_names, desc="Tokenizing for prediction")
    model_inputs = feats.remove_columns(["example_id", "offset_mapping"])
    start_logits, end_logits = trainer.predict(model_inputs).predictions[:2]
    return best_spans(examples, feats, start_logits, end_logits, cfg)


# -------------------------------------------------------------------- Training
def train(model_name: str, data, cfg: dict) -> Dict:
    set_seed(cfg["seed"])
    s = training_settings(cfg, TASK, model_name)
    out = output_dir(cfg, TASK, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    cols = data["train"].column_names
    train_feats = data["train"].map(lambda b: prepare_train_features(b, tokenizer, cfg), batched=True,
                                    remove_columns=cols, desc="Tokenizing train")
    tune_feats = data["tune"].map(lambda b: prepare_train_features(b, tokenizer, cfg), batched=True,
                                  remove_columns=cols, desc="Tokenizing tune")

    args = TrainingArguments(
        output_dir=str(out),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    trainer = Trainer(model=model, args=args, train_dataset=train_feats, eval_dataset=tune_feats,
                      data_collator=default_data_collator, processing_class=tokenizer)
    t0 = time.time()
    trainer.train()
    train_minutes = round((time.time() - t0) / 60, 1)

    best_dir = out / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    for ckpt in out.glob("checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)

    # Threshold chosen on tune, then applied unchanged to test.
    tune_spans = predict_spans(trainer, tokenizer, data["tune"], cfg)
    threshold, tune_f1 = tune_threshold(tune_spans, references(data["tune"]))
    test_spans = predict_spans(trainer, tokenizer, data["test"], cfg)
    test_refs = references(data["test"])
    preds = apply_threshold(test_spans, threshold)
    metrics = qa_metrics(preds, test_refs)
    metrics["f1_threshold_0"] = qa_metrics(apply_threshold(test_spans, 0.0), test_refs)["f1"]
    metrics["no_answer_threshold"] = round(threshold, 3)
    metrics["tune_f1"] = tune_f1
    metrics["train_minutes"] = train_minutes

    results_dir = resolve(cfg["paths"]["results_dir"])
    sample_rows = []
    for ex in data["test"].select(range(min(200, len(data["test"])))):
        gold = list(ex["answers"]["text"])
        pred = preds[ex["id"]]
        sample_rows.append({"id": ex["id"], "question": ex["question"], "context": ex["context"],
                            "gold": gold, "prediction": pred,
                            "correct": qa_metrics({ex["id"]: pred}, {ex["id"]: gold})["exact_match"] == 100})
    write_jsonl(sample_rows, results_dir / f"{TASK}_{safe_name(model_name)}_samples.jsonl")
    with open(results_dir / f"{TASK}_{safe_name(model_name)}_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    save_results(cfg, TASK, model_name, metrics, extra={
        "settings": s, "data_sizes": {k: len(v) for k, v in data.items()},
        "max_seq_len": cfg[TASK]["max_seq_len"], "doc_stride": cfg[TASK]["doc_stride"],
        "checkpoint": str(best_dir),
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
    data = load_qa(cfg, args.max_samples)
    for name in (cfg[TASK]["models"] if args.model == "all" else [args.model]):
        print(f"\n===== Training {name} =====")
        train(name, data, cfg)


if __name__ == "__main__":
    main()
