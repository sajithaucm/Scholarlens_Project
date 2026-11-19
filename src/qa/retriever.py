"""Passage retrieval for question answering over a long paper.

A paper is far longer than a QA model's 384-token window, so we first split
it into overlapping ~120-word passages and retrieve the top-k most relevant
ones for the question. Two retrievers are compared:
  * dense:  sentence-transformers (all-MiniLM-L6-v2) cosine similarity
  * sparse: TF-IDF cosine similarity (classic IR baseline)

Evaluation (no extra labels needed): for every answerable SQuAD 2.0 dev
question we retrieve among ALL passages built from that Wikipedia article and
check whether a retrieved passage contains a gold answer -> Recall@1, Recall@k, MRR.

Usage:
    python -m src.qa.retriever --evaluate             # both retrievers from config
    python -m src.qa.retriever --evaluate --max-questions 500
"""

import argparse
import collections
import time
from typing import List, Optional

import numpy as np

from src.evaluation.metrics import retrieval_metrics
from src.preprocessing import split_into_passages
from src.utils import load_config, save_results


class Retriever:
    """Index a list of passages and return the top-k for a query."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.passages: List[str] = []
        if model_name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.encoder = None
        else:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name, device=device)

    def index(self, passages: List[str]) -> "Retriever":
        self.passages = list(passages)
        if self.encoder is None:
            self.matrix = self.vectorizer.fit_transform(self.passages)
        else:
            self.matrix = self.encoder.encode(self.passages, normalize_embeddings=True,
                                              convert_to_numpy=True, show_progress_bar=False)
        return self

    def scores(self, query: str) -> np.ndarray:
        if self.encoder is None:
            q = self.vectorizer.transform([query])
            return (self.matrix @ q.T).toarray().ravel()
        q = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.matrix @ q

    def search(self, query: str, top_k: int = 3):
        """Return [(passage, score)] sorted best first."""
        s = self.scores(query)
        order = np.argsort(-s)[:top_k]
        return [(self.passages[i], float(s[i])) for i in order]


def evaluate_retrieval(cfg: dict, model_name: str, max_questions: Optional[int] = None) -> dict:
    """Recall@1 / Recall@k / MRR on SQuAD 2.0 dev, retrieving within each article."""
    from datasets import load_dataset

    rc = cfg["retrieval"]
    k = rc["top_k_passages"]
    dev = load_dataset(cfg["qa"]["dataset"], split="validation")
    by_title = collections.defaultdict(list)
    for ex in dev:
        by_title[ex["title"]].append(ex)

    retriever = Retriever(model_name)
    hits, n = [], 0
    t0 = time.time()
    for title, exs in by_title.items():
        contexts = list(dict.fromkeys(ex["context"] for ex in exs))   # unique paragraphs, in order
        passages = split_into_passages("\n".join(contexts), rc["passage_max_words"],
                                       rc["passage_overlap_sentences"])
        retriever.index(passages)
        norm_passages = [" ".join(p.lower().split()) for p in passages]
        for ex in exs:
            if not ex["answers"]["text"]:
                continue                     # unanswerable: nothing to retrieve
            order = np.argsort(-retriever.scores(ex["question"]))[:max(k, 10)]
            golds = [" ".join(a.lower().split()) for a in ex["answers"]["text"]]
            hits.append([any(g in norm_passages[i] for g in golds) for i in order])
            n += 1
            if max_questions and n >= max_questions:
                break
        if max_questions and n >= max_questions:
            break
    metrics = retrieval_metrics(hits, k)
    metrics["seconds"] = round(time.time() - t0, 1)
    save_results(cfg, "retrieval", model_name, metrics, extra={"settings": rc})
    print(model_name, metrics)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--model", default="all")
    p.add_argument("--max-questions", type=int, default=None)
    args = p.parse_args()
    cfg = load_config()
    if args.evaluate:
        models = cfg["retrieval"]["models"] if args.model == "all" else [args.model]
        for m in models:
            evaluate_retrieval(cfg, m, args.max_questions)


if __name__ == "__main__":
    main()
