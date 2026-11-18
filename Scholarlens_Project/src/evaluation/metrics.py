"""Evaluation metrics shared by every ScholarLens module.

All three tasks report at least three metrics (course requirement):
  * Summarization: ROUGE-1, ROUGE-2, ROUGE-L(sum), BERTScore-F1, average length
  * QA:            Exact Match, F1, separately for answerable / unanswerable questions
  * NER:           entity-level precision, recall, F1 (micro) + per-type F1
  * Retrieval:     Recall@1, Recall@k, Mean Reciprocal Rank

The SQuAD EM/F1 code follows the official SQuAD 2.0 evaluation script so our
numbers are comparable to published results. Everything here runs offline
except BERTScore, which downloads its scoring model the first time.
"""

import collections
import re
import string
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------- Summarization
def _rouge_lsum_text(text: str) -> str:
    """rougeLsum expects one sentence per line."""
    from src.preprocessing import split_sentences
    return "\n".join(split_sentences(text)) or text


def summarization_metrics(predictions: Sequence[str], references: Sequence[str],
                          use_bertscore: bool = True, bertscore_model: Optional[str] = None,
                          batch_size: int = 16) -> Dict[str, float]:
    """ROUGE-1/2/L/Lsum (F-measure x100), BERTScore F1 (x100) and length stats."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    totals = collections.defaultdict(list)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(_rouge_lsum_text(ref), _rouge_lsum_text(pred))
        for key, val in scores.items():
            totals[key].append(val.fmeasure)
    out = {k: round(100 * float(np.mean(v)), 2) for k, v in totals.items()}
    out["avg_pred_words"] = round(float(np.mean([len(p.split()) for p in predictions])), 1)
    out["avg_ref_words"] = round(float(np.mean([len(r.split()) for r in references])), 1)

    if use_bertscore:
        from bert_score import score as bert_score
        kwargs = {"lang": "en", "batch_size": batch_size, "verbose": False}
        if bertscore_model:
            kwargs["model_type"] = bertscore_model
        _, _, f1 = bert_score(list(predictions), list(references), **kwargs)
        out["bertscore_f1"] = round(100 * float(f1.mean()), 2)
    return out


# --------------------------------------------------------------------------- QA
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace (official SQuAD)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _tokens(s: str) -> List[str]:
    return normalize_answer(s).split() if s else []


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:  # unanswerable: correct only if both are empty
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def qa_metrics(predictions: Dict[str, str], references: Dict[str, List[str]]) -> Dict[str, float]:
    """SQuAD 2.0 metrics.

    predictions: {question_id: predicted answer text ("" = no answer)}
    references:  {question_id: list of gold answers ([] = unanswerable)}
    """
    em, f1, has_ans, no_ans = [], [], [], []
    for qid, golds in references.items():
        pred = predictions.get(qid, "")
        golds = golds or [""]
        e = max(exact_match(pred, g) for g in golds)
        f = max(token_f1(pred, g) for g in golds)
        em.append(e)
        f1.append(f)
        (has_ans if golds != [""] else no_ans).append((e, f))

    def avg(xs, i):
        return round(100 * float(np.mean([x[i] for x in xs])), 2) if xs else None

    return {
        "exact_match": round(100 * float(np.mean(em)), 2),
        "f1": round(100 * float(np.mean(f1)), 2),
        "has_ans_exact_match": avg(has_ans, 0),
        "has_ans_f1": avg(has_ans, 1),
        "no_ans_accuracy": avg(no_ans, 0),
        "n_total": len(em),
        "n_has_ans": len(has_ans),
        "n_no_ans": len(no_ans),
    }


# -------------------------------------------------------------------------- NER
def ner_metrics(true_tags: List[List[str]], pred_tags: List[List[str]]) -> Dict[str, object]:
    """Entity-level (exact span + type) precision/recall/F1 using seqeval."""
    from seqeval.metrics import (accuracy_score, classification_report, f1_score,
                                 precision_score, recall_score)

    report = classification_report(true_tags, pred_tags, output_dict=True, zero_division=0)
    per_type = {k: round(100 * v["f1-score"], 2) for k, v in report.items()
                if k not in ("micro avg", "macro avg", "weighted avg")}
    return {
        "precision": round(100 * precision_score(true_tags, pred_tags, zero_division=0), 2),
        "recall": round(100 * recall_score(true_tags, pred_tags, zero_division=0), 2),
        "f1": round(100 * f1_score(true_tags, pred_tags, zero_division=0), 2),
        "macro_f1": round(100 * f1_score(true_tags, pred_tags, average="macro", zero_division=0), 2),
        "token_accuracy": round(100 * accuracy_score(true_tags, pred_tags), 2),
        "per_type_f1": per_type,
    }


# -------------------------------------------------------------------- Retrieval
def retrieval_metrics(ranked_hits: List[List[bool]], k: int) -> Dict[str, float]:
    """ranked_hits[i][r] is True if the r-th ranked passage for query i contains the answer."""
    r1 = [any(h[:1]) for h in ranked_hits]
    rk = [any(h[:k]) for h in ranked_hits]
    rr = [next((1.0 / (r + 1) for r, hit in enumerate(h) if hit), 0.0) for h in ranked_hits]
    return {
        "recall_at_1": round(100 * float(np.mean(r1)), 2),
        f"recall_at_{k}": round(100 * float(np.mean(rk)), 2),
        "mrr": round(100 * float(np.mean(rr)), 2),
        "n_queries": len(ranked_hits),
    }
