"""End-to-end ScholarLens pipeline: paper text in -> summary, entities, answers out.

    from src.pipeline import ScholarLens
    lens = ScholarLens.from_config()
    lens.load_paper(text)
    lens.summarize()
    lens.extract_entities()
    lens.answer("Which dataset do the authors use?")

Each stage loads the fine-tuned checkpoint named in config.yaml (demo:
section). If a checkpoint folder does not exist yet, the base model is used
instead and `self.warnings` says so, so the demo never silently pretends a
base model is fine-tuned.
"""

import collections
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
                          AutoModelForTokenClassification, AutoTokenizer)

from src.data_loading import _TOKEN_RE
from src.preprocessing import clean_text, split_into_passages, split_sentences
from src.qa.retriever import Retriever
from src.utils import load_config, resolve


def _model_source(path_or_id: str, fallback: str, warnings: List[str], what: str, out_dir: Path) -> str:
    """Find a model: <output_dir>/<path>, then <repo>/<path>, then a Hub ID, else the base model."""
    for candidate in (out_dir / path_or_id, resolve(path_or_id)):
        if candidate.exists():
            return str(candidate)
    if path_or_id.count("/") == 1 and not path_or_id.endswith("/best"):
        return path_or_id                      # looks like a Hugging Face Hub ID (user/model)
    warnings.append(f"{what}: fine-tuned checkpoint '{path_or_id}' not found, using '{fallback}' instead.")
    return fallback


class ScholarLens:
    def __init__(self, cfg: dict, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.warnings: List[str] = []
        d, out = cfg["demo"], resolve(cfg["paths"]["output_dir"])
        self.sum_src = _model_source(d["summarization_model"], cfg["summarization"]["models"][0],
                                     self.warnings, "Summarization", out)
        self.qa_src = _model_source(d["qa_model"], "deepset/roberta-base-squad2", self.warnings, "QA", out)
        self.ner_src = _model_source(d["ner_model"], cfg["ner"]["models"][0], self.warnings, "NER", out)
        self._sum = self._qa = self._ner = None
        self.retriever = Retriever(d.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2"),
                                   device=self.device)
        self.text = ""
        self.passages: List[str] = []

    @classmethod
    def from_config(cls, path: Optional[Path] = None, device: Optional[str] = None) -> "ScholarLens":
        return cls(load_config(path) if path else load_config(), device)

    # ------------------------------------------------------------- loading
    def _lazy(self, kind: str):
        if kind == "sum" and self._sum is None:
            tok = AutoTokenizer.from_pretrained(self.sum_src)
            self._sum = (tok, AutoModelForSeq2SeqLM.from_pretrained(self.sum_src).to(self.device).eval())
        if kind == "qa" and self._qa is None:
            tok = AutoTokenizer.from_pretrained(self.qa_src)
            self._qa = (tok, AutoModelForQuestionAnswering.from_pretrained(self.qa_src).to(self.device).eval())
        if kind == "ner" and self._ner is None:
            tok = AutoTokenizer.from_pretrained(self.ner_src)
            model = AutoModelForTokenClassification.from_pretrained(self.ner_src).to(self.device).eval()
            if len(model.config.id2label) <= 2:
                self.warnings.append("NER model has no trained label head yet; entity output is not meaningful.")
            self._ner = (tok, model)
        return {"sum": self._sum, "qa": self._qa, "ner": self._ner}[kind]

    def load_paper(self, text: str) -> int:
        """Clean the paper and build the passage index. Returns the number of passages."""
        self.text = clean_text(text)
        rc = self.cfg["retrieval"]
        self.passages = split_into_passages(self.text, rc["passage_max_words"], rc["passage_overlap_sentences"])
        if self.passages:
            self.retriever.index(self.passages)
        return len(self.passages)

    # ------------------------------------------------------------- summarization
    @torch.no_grad()
    def _summarize_one(self, text: str, max_new_tokens: Optional[int] = None) -> str:
        tok, model = self._lazy("sum")
        c = self.cfg["summarization"]
        prefix = c.get("t5_prefix", "summarize: ") if "t5" in self.sum_src.lower() else ""
        gen = dict(c.get("generation", {}))
        if max_new_tokens:
            gen["max_new_tokens"] = max_new_tokens
            gen["min_new_tokens"] = min(gen.get("min_new_tokens", 0), max_new_tokens // 2)
        enc = tok(prefix + text, max_length=c["max_input_tokens"], truncation=True,
                  return_tensors="pt").to(self.device)
        return tok.decode(model.generate(**enc, **gen)[0], skip_special_tokens=True).strip()

    def summarize(self, strategy: Optional[str] = None) -> str:
        """'truncate': summarize the first max_input_tokens tokens (matches training).
        'chunked': summarize ~700-word chunks, then summarize the joined chunk summaries."""
        strategy = strategy or self.cfg["demo"].get("summary_strategy", "truncate")
        if not self.text:
            return ""
        if strategy == "truncate":
            return self._summarize_one(self.text)
        sents, chunks, cur = split_sentences(self.text), [], []
        for s in sents:
            cur.append(s)
            if sum(len(x.split()) for x in cur) > 700:
                chunks.append(" ".join(cur))
                cur = []
        if cur:
            chunks.append(" ".join(cur))
        chunks = chunks[:8]                          # cap runtime for very long papers
        if len(chunks) == 1:
            return self._summarize_one(chunks[0])
        partial = [self._summarize_one(c, max_new_tokens=96) for c in chunks]
        return self._summarize_one(" ".join(partial))

    # ------------------------------------------------------------- NER
    @torch.no_grad()
    def extract_entities(self, max_sentences: Optional[int] = None) -> Dict[str, List[str]]:
        """Return {entity type: [unique entity strings]} found in the paper."""
        tok, model = self._lazy("ner")
        id2label = model.config.id2label
        max_sentences = max_sentences or self.cfg["demo"].get("max_ner_sentences", 200)
        found = collections.defaultdict(collections.Counter)
        for sent in split_sentences(self.text)[:max_sentences]:
            words = _TOKEN_RE.findall(sent)   # same word splitting as the SciERC fallback loader
            if not words:
                continue
            enc = tok(words, is_split_into_words=True, truncation=True,
                      max_length=self.cfg["ner"]["max_seq_len"], return_tensors="pt")
            word_ids = enc.word_ids()
            pred = model(**enc.to(self.device)).logits[0].argmax(-1).tolist()
            tags, prev = ["O"] * len(words), None
            for pos, wid in enumerate(word_ids):
                if wid is not None and wid != prev:
                    tags[wid] = id2label[pred[pos]]
                prev = wid
            for typ, span in bio_to_spans(words, tags):
                found[typ][span.strip(".,;:()")] += 1
        return {t: [e for e, _ in c.most_common(25) if e] for t, c in sorted(found.items())}

    def tag_text(self, text: str):
        """Word-level (word, label) pairs for highlighting one passage in the demo."""
        tok, model = self._lazy("ner")
        words = _TOKEN_RE.findall(text)
        if not words:
            return []
        enc = tok(words, is_split_into_words=True, truncation=True,
                  max_length=self.cfg["ner"]["max_seq_len"], return_tensors="pt")
        word_ids = enc.word_ids()
        with torch.no_grad():
            pred = model(**enc.to(self.device)).logits[0].argmax(-1).tolist()
        tags, prev = ["O"] * len(words), None
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                tags[wid] = model.config.id2label[pred[pos]]
            prev = wid
        return [(w, None if t == "O" else t[2:]) for w, t in zip(words, tags)]

    # ------------------------------------------------------------- QA
    @torch.no_grad()
    def answer(self, question: str, top_k: Optional[int] = None, max_answer_tokens: int = 30) -> Dict:
        """Retrieve top-k passages, run extractive QA on each, return the best span."""
        tok, model = self._lazy("qa")
        top_k = top_k or self.cfg["retrieval"]["top_k_passages"]
        if not self.passages:
            return {"answer": "", "score": 0.0, "passage": "", "retrieved": []}
        retrieved = self.retriever.search(question, top_k)
        best = {"answer": "", "score": float("-inf"), "passage": ""}
        for passage, _ in retrieved:
            enc = tok(question, passage, truncation="only_second", max_length=self.cfg["qa"]["max_seq_len"],
                      return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            seq_ids = enc.sequence_ids(0)
            out = model(**enc.to(self.device))
            s_log, e_log = out.start_logits[0], out.end_logits[0]
            null = float(s_log[0] + e_log[0])
            ctx = [i for i, sid in enumerate(seq_ids) if sid == 1]
            s_top = sorted(ctx, key=lambda i: -float(s_log[i]))[:20]
            e_top = sorted(ctx, key=lambda i: -float(e_log[i]))[:20]
            for s in s_top:
                for e in e_top:
                    if e < s or e - s + 1 > max_answer_tokens:
                        continue
                    score = float(s_log[s] + e_log[e]) - null
                    if score > best["score"]:
                        best = {"answer": passage[offsets[s][0]:offsets[e][1]], "score": score, "passage": passage}
        if best["score"] <= self.cfg["qa"].get("no_answer_threshold", 0.0):
            best["answer"] = ""                       # model prefers "no answer"
        best["retrieved"] = [{"passage": p, "similarity": round(sc, 3)} for p, sc in retrieved]
        best["score"] = round(best["score"], 3) if best["score"] != float("-inf") else None
        return best


def bio_to_spans(words: List[str], tags: List[str]):
    """[(type, 'entity text')] from word-level BIO tags (an I- without B- starts a new entity)."""
    spans, cur_type, cur = [], None, []
    for w, t in zip(words, tags):
        if t.startswith("B-") or (t.startswith("I-") and t[2:] != cur_type):
            if cur:
                spans.append((cur_type, " ".join(cur)))
            cur_type, cur = t[2:], [w]
        elif t.startswith("I-"):
            cur.append(w)
        else:
            if cur:
                spans.append((cur_type, " ".join(cur)))
            cur_type, cur = None, []
    if cur:
        spans.append((cur_type, " ".join(cur)))
    return spans
