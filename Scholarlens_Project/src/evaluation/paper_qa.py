"""Evaluate the full pipeline (retrieval + QA) on questions about real papers.

SQuAD is Wikipedia text; the demo answers questions about research papers.
This script measures that domain shift on a small hand-written test set:

    eval/papers/<paper_id>.txt     plain text of each paper (copy from the PDF)
    eval/paper_qa.jsonl            one question per line:
        {"paper_id": "attention", "question": "...", "answers": ["..."]}
        (use "answers": [] for a question the paper does not answer)

Usage:
    python -m src.evaluation.paper_qa
    python -m src.evaluation.paper_qa --qa-model outputs/qa/distilbert-base-uncased/best
"""

import argparse
from pathlib import Path

from src.evaluation.metrics import normalize_answer, qa_metrics
from src.pipeline import ScholarLens
from src.utils import ROOT, load_config, read_jsonl, resolve, safe_name, save_results, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=str(ROOT / "eval" / "paper_qa.jsonl"))
    p.add_argument("--qa-model", default=None, help="override demo.qa_model from config")
    args = p.parse_args()

    cfg = load_config()
    if args.qa_model:
        cfg["demo"]["qa_model"] = args.qa_model
    if not Path(args.file).exists():
        raise SystemExit(f"{args.file} not found. Write your questions first (see eval/README.md and "
                         "eval/paper_qa.example.jsonl).")
    rows = read_jsonl(args.file)
    lens = ScholarLens(cfg)
    for w in lens.warnings:
        print("WARNING:", w)

    preds, refs, out_rows, hits, current = {}, {}, [], [], None
    for i, row in enumerate(rows):
        if row["paper_id"] != current:
            lens.load_paper((ROOT / "eval" / "papers" / f"{row['paper_id']}.txt").read_text())
            current = row["paper_id"]
        res = lens.answer(row["question"])
        qid = f"q{i}"
        preds[qid], refs[qid] = res["answer"], row["answers"]
        if row["answers"]:
            golds = [normalize_answer(a) for a in row["answers"]]
            hits.append(any(g in normalize_answer(r["passage"]) for r in res["retrieved"] for g in golds))
        out_rows.append({**row, "prediction": res["answer"], "score": res["score"],
                         "top_passage": res["retrieved"][0]["passage"] if res["retrieved"] else ""})

    metrics = qa_metrics(preds, refs)
    metrics["retrieval_recall_at_k"] = round(100 * sum(hits) / len(hits), 2) if hits else None
    src = Path(lens.qa_src)
    qa_name = src.parent.name if src.name == "best" else lens.qa_src   # outputs/qa/<model>/best -> <model>
    name = f"pipeline-{qa_name}"
    save_results(cfg, "paper_qa", name, metrics, extra={"n_questions": len(rows)})
    write_jsonl(out_rows, resolve(cfg["paths"]["results_dir"]) / f"paper_qa_{safe_name(name)}_samples.jsonl")
    print(metrics)


if __name__ == "__main__":
    main()
