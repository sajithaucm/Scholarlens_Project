"""Simple non-neural baselines, scored with the same metrics as the transformers.

A fine-tuned model is only impressive if it beats something simple, so every
results table in the report starts with these rows:

  * Summarization: Lead-k (first k sentences) and TextRank (graph-based extractive)
  * QA:            "always no-answer" and TF-IDF best-sentence answer
  * NER:           dictionary lookup (longest match of entity strings seen in training)

Usage:
    python -m src.baselines --task summarization
    python -m src.baselines --task qa
    python -m src.baselines --task ner
    python -m src.baselines --task all --max-samples 200     # quick test
"""

import argparse
import collections
from typing import Dict, List

import numpy as np

from src.evaluation.metrics import ner_metrics, qa_metrics, summarization_metrics
from src.preprocessing import split_sentences
from src.utils import load_config, save_results


# ---------------------------------------------------------------- Summarization
def lead_k(article: str, k: int = 3) -> str:
    return " ".join(split_sentences(article)[:k])


def textrank(article: str, k: int = 6, max_sentences: int = 200, damping: float = 0.85) -> str:
    """Rank sentences by PageRank over a TF-IDF cosine-similarity graph; keep top-k in original order."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sents = [s for s in split_sentences(article) if len(s.split()) >= 5][:max_sentences]
    if len(sents) <= k:
        return " ".join(sents)
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:              # only stop words
        return " ".join(sents[:k])
    sim = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True)
    trans = np.divide(sim, row_sums, out=np.full_like(sim, 1.0 / len(sents)), where=row_sums > 0)
    scores = np.full(len(sents), 1.0 / len(sents))
    for _ in range(50):             # power iteration
        new = (1 - damping) / len(sents) + damping * trans.T @ scores
        if np.abs(new - scores).sum() < 1e-6:
            break
        scores = new
    top = sorted(np.argsort(-scores)[:k])
    return " ".join(sents[i] for i in top)


def run_summarization(cfg: dict, max_samples=None, use_bertscore=True) -> Dict[str, dict]:
    from src.data_loading import load_summarization
    test = load_summarization(cfg, max_samples)["test"]
    refs = test["abstract"]
    results = {}
    for name, fn in (("lead-3", lambda a: lead_k(a, 3)), ("textrank", textrank)):
        preds = [fn(a) for a in test["article"]]
        results[name] = summarization_metrics(preds, refs, use_bertscore=use_bertscore)
        results[name]["n_test"] = len(preds)
        save_results(cfg, "summarization", f"baseline-{name}", results[name])
    return results


# --------------------------------------------------------------------------- QA
def tfidf_sentence_answer(question: str, context: str) -> str:
    """Return the context sentence most similar to the question (never abstains)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = split_sentences(context) or [context]
    try:
        vec = TfidfVectorizer(stop_words="english").fit(sents + [question])
    except ValueError:
        return sents[0]
    sim = (vec.transform(sents) @ vec.transform([question]).T).toarray().ravel()
    return sents[int(sim.argmax())]


def run_qa(cfg: dict, max_samples=None) -> Dict[str, dict]:
    from src.data_loading import load_qa
    test = load_qa(cfg, max_samples)["test"]
    refs = {ex["id"]: list(ex["answers"]["text"]) for ex in test}
    results = {
        "always-no-answer": qa_metrics({qid: "" for qid in refs}, refs),
        "tfidf-sentence": qa_metrics({ex["id"]: tfidf_sentence_answer(ex["question"], ex["context"])
                                      for ex in test}, refs),
    }
    for name, m in results.items():
        save_results(cfg, "qa", f"baseline-{name}", m)
    return results


# -------------------------------------------------------------------------- NER
def build_gazetteer(dataset, label_names: List[str]) -> Dict[tuple, str]:
    """Map every lower-cased entity string in training to its most frequent type."""
    counts = collections.defaultdict(collections.Counter)
    for tokens, tags in zip(dataset["tokens"], dataset["ner_tags"]):
        tags = [label_names[t] for t in tags]
        i = 0
        while i < len(tags):
            if tags[i].startswith("B-"):
                typ, j = tags[i][2:], i + 1
                while j < len(tags) and tags[j] == f"I-{typ}":
                    j += 1
                counts[tuple(t.lower() for t in tokens[i:j])][typ] += 1
                i = j
            else:
                i += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


def gazetteer_tag(tokens: List[str], gaz: Dict[tuple, str], max_len: int = 8) -> List[str]:
    low = [t.lower() for t in tokens]
    tags, i = ["O"] * len(tokens), 0
    while i < len(tokens):
        for n in range(min(max_len, len(tokens) - i), 0, -1):      # longest match first
            typ = gaz.get(tuple(low[i:i + n]))
            if typ:
                tags[i] = f"B-{typ}"
                for j in range(i + 1, i + n):
                    tags[j] = f"I-{typ}"
                i += n
                break
        else:
            i += 1
    return tags


def run_ner(cfg: dict, max_samples=None) -> Dict[str, dict]:
    from src.data_loading import load_ner
    data = load_ner(cfg, max_samples)
    names = data["train"].features["ner_tags"].feature.names
    gaz = build_gazetteer(data["train"], names)
    true = [[names[t] for t in tags] for tags in data["test"]["ner_tags"]]
    pred = [gazetteer_tag(toks, gaz) for toks in data["test"]["tokens"]]
    m = ner_metrics(true, pred)
    m["gazetteer_entries"] = len(gaz)
    save_results(cfg, "ner", "baseline-dictionary", m)
    return {"dictionary": m}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["summarization", "qa", "ner", "all"], required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-bertscore", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    if args.task in ("summarization", "all"):
        print(run_summarization(cfg, args.max_samples, not args.no_bertscore))
    if args.task in ("qa", "all"):
        print(run_qa(cfg, args.max_samples))
    if args.task in ("ner", "all"):
        print(run_ner(cfg, args.max_samples))


if __name__ == "__main__":
    main()
