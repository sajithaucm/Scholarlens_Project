"""Fine-tune and evaluate abstractive summarizers (BART vs T5) on arXiv papers.

Input: cleaned paper body (`article`), truncated to max_input_tokens.
Target: the paper's abstract.

Usage (Colab, from the repo root):
    python -m src.summarization.train --model facebook/bart-base
    python -m src.summarization.train --model all                 # both models in config
    python -m src.summarization.train --model t5-small --max-samples 200 --epochs 1   # quick test

Outputs:
    outputs/summarization/<model>/best/          fine-tuned model + tokenizer
    results/summarization_<model>.json           test metrics + settings
    results/summarization_<model>_history.json   training/validation loss per step
    results/summarization_<model>_samples.jsonl  example predictions for error analysis
"""

import argparse
import json
import shutil
import time
from typing import Dict, List

import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

from src.data_loading import load_summarization
from src.evaluation.metrics import summarization_metrics
from src.utils import (load_config, output_dir, resolve, safe_name, save_results, set_seed,
                       training_settings, use_fp16, warmup_kwargs, write_jsonl)

TASK = "summarization"


def input_prefix(cfg: dict, model_name: str) -> str:
    """T5 was pre-trained with task prefixes; BART needs none."""
    return cfg[TASK].get("t5_prefix", "summarize: ") if "t5" in model_name.lower() else ""


def tokenize_dataset(data, tokenizer, cfg: dict, model_name: str):
    """Tokenize articles (inputs) and abstracts (labels), truncating both."""
    c, prefix = cfg[TASK], input_prefix(cfg, model_name)

    def tok(batch):
        enc = tokenizer([prefix + a for a in batch["article"]],
                        max_length=c["max_input_tokens"], truncation=True)
        labels = tokenizer(text_target=batch["abstract"], max_length=c["max_target_tokens"], truncation=True)
        enc["labels"] = labels["input_ids"]
        return enc

    return data.map(tok, batched=True, remove_columns=data["train"].column_names, desc="Tokenizing")


@torch.no_grad()
def generate_summaries(model, tokenizer, articles: List[str], cfg: dict, model_name: str,
                       batch_size: int = 8) -> List[str]:
    """Beam-search summaries for a list of articles (sorted by length for speed)."""
    c, prefix = cfg[TASK], input_prefix(cfg, model_name)
    gen = dict(c.get("generation", {}))
    device = next(model.parameters()).device
    model.eval()
    order = sorted(range(len(articles)), key=lambda i: -len(articles[i]))
    outputs = [""] * len(articles)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([prefix + articles[i] for i in idx], max_length=c["max_input_tokens"],
                        truncation=True, padding=True, return_tensors="pt").to(device)
        ids = model.generate(**enc, **gen)
        for i, text in zip(idx, tokenizer.batch_decode(ids, skip_special_tokens=True)):
            outputs[i] = text.strip()
    return outputs


def train(model_name: str, data, cfg: dict, use_bertscore: bool = True) -> Dict:
    """Fine-tune one model, evaluate it on the test split and save everything."""
    set_seed(cfg["seed"])
    s = training_settings(cfg, TASK, model_name)
    out = output_dir(cfg, TASK, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenized = tokenize_dataset(data, tokenizer, cfg, model_name)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,          # best = lowest validation loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        learning_rate=float(s["learning_rate"]),
        per_device_train_batch_size=int(s["batch_size"]),
        per_device_eval_batch_size=int(s["batch_size"]),
        gradient_accumulation_steps=int(s.get("grad_accum_steps", 1)),
        num_train_epochs=float(s["epochs"]),
        weight_decay=float(s["weight_decay"]),
        **warmup_kwargs(float(s.get("warmup_ratio", 0.0))),
        fp16=use_fp16(s, model_name),
        logging_steps=int(s.get("logging_steps", 50)),
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    t0 = time.time()
    trainer.train()
    train_minutes = round((time.time() - t0) / 60, 1)

    best_dir = out / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    for ckpt in out.glob("checkpoint-*"):          # keep only the best model (saves Drive space)
        shutil.rmtree(ckpt, ignore_errors=True)

    # ---- Test-set evaluation (generation)
    t0 = time.time()
    preds = generate_summaries(trainer.model, tokenizer, data["test"]["article"], cfg, model_name)
    metrics = summarization_metrics(preds, data["test"]["abstract"], use_bertscore=use_bertscore)
    metrics["train_minutes"] = train_minutes
    metrics["generate_minutes"] = round((time.time() - t0) / 60, 1)
    metrics["n_test"] = len(preds)

    results_dir = resolve(cfg["paths"]["results_dir"])
    write_jsonl(
        [{"article_start": a[:1500], "reference": r, "prediction": p}
         for a, r, p in zip(data["test"]["article"][:30], data["test"]["abstract"][:30], preds[:30])],
        results_dir / f"{TASK}_{safe_name(model_name)}_samples.jsonl",
    )
    with open(results_dir / f"{TASK}_{safe_name(model_name)}_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    save_results(cfg, TASK, model_name, metrics, extra={
        "settings": s,
        "data_sizes": {k: len(v) for k, v in data.items()},
        "max_input_tokens": cfg[TASK]["max_input_tokens"],
        "generation": cfg[TASK].get("generation", {}),
        "checkpoint": str(best_dir),
    })
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", help="model name from config, or 'all'")
    p.add_argument("--max-samples", type=int, default=None, help="cap every split (quick tests)")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--no-bertscore", action="store_true", help="skip BERTScore (faster)")
    args = p.parse_args()

    cfg = load_config()
    if args.epochs is not None:
        cfg[TASK].setdefault("training", {})["epochs"] = args.epochs
    data = load_summarization(cfg, args.max_samples)
    models = cfg[TASK]["models"] if args.model == "all" else [args.model]
    for name in models:
        print(f"\n===== Training {name} =====")
        train(name, data, cfg, use_bertscore=not args.no_bertscore)


if __name__ == "__main__":
    main()
