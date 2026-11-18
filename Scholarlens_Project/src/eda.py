"""Exploratory data analysis helpers used by notebooks/00_setup_and_eda.ipynb.

Produces the dataset statistics table and figures for the report's Datasets
section, including the sentence count that shows the >= 50,000-sentence
requirement is met.
"""

import collections
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.preprocessing import split_sentences

TEXT, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def summarization_stats(data, sentence_sample: int = 500) -> Dict[str, object]:
    """Word/sentence statistics for arXiv articles and abstracts."""
    rows = {}
    for split, ds in data.items():
        art = [len(a.split()) for a in ds["article"]]
        abs_ = [len(a.split()) for a in ds["abstract"]]
        sample = ds["article"][:sentence_sample]
        sents_per_doc = float(np.mean([len(split_sentences(a)) for a in sample])) if sample else 0.0
        rows[split] = {"papers": len(ds), "article_words_mean": round(float(np.mean(art)), 1),
                       "article_words_median": float(np.median(art)),
                       "abstract_words_mean": round(float(np.mean(abs_)), 1),
                       "sentences_per_article": round(sents_per_doc, 1),
                       "est_total_sentences": int(sents_per_doc * len(ds))}
    return rows


def qa_stats(data) -> Dict[str, object]:
    rows = {}
    for split, ds in data.items():
        n_ans = sum(1 for a in ds["answers"] if a["text"])
        rows[split] = {"questions": len(ds), "answerable": n_ans, "unanswerable": len(ds) - n_ans,
                       "unique_contexts": len(set(ds["context"])),
                       "context_words_mean": round(float(np.mean([len(c.split()) for c in ds["context"]])), 1),
                       "answer_words_mean": round(float(np.mean([len(a["text"][0].split())
                                                                 for a in ds["answers"] if a["text"]] or [0])), 1)}
    return rows


def ner_stats(data) -> Dict[str, object]:
    names = data["train"].features["ner_tags"].feature.names
    rows = {}
    for split, ds in data.items():
        counts = collections.Counter(names[t][2:] for tags in ds["ner_tags"] for t in tags
                                     if names[t].startswith("B-"))
        rows[split] = {"sentences": len(ds), "tokens": sum(len(t) for t in ds["tokens"]),
                       "entities": sum(counts.values()), **{f"n_{k}": v for k, v in sorted(counts.items())}}
    return rows


def to_frame(stats: Dict[str, Dict]) -> pd.DataFrame:
    return pd.DataFrame(stats).T


def plot_length_hist(values_a, values_b, label_a, label_b, title, path: Path, xlabel="words"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=200)
    for ax, vals, lab, color in ((axes[0], values_a, label_a, BLUE), (axes[1], values_b, label_b, ORANGE)):
        ax.hist(vals, bins=40, color=color, edgecolor="white", linewidth=0.6)
        ax.axvline(np.median(vals), color=TEXT, linewidth=1, linestyle="--")
        ax.text(np.median(vals), ax.get_ylim()[1] * 0.92, f" median {np.median(vals):.0f}", color=TEXT, fontsize=8)
        _style(ax, lab, xlabel, "count")
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_entity_counts(stats: Dict[str, Dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train = {k[2:]: v for k, v in stats["train"].items() if k.startswith("n_")}
    items = sorted(train.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=200)
    ax.bar([k for k, _ in items], [v for _, v in items], color=BLUE, width=0.6, edgecolor="white")
    for i, (_, v) in enumerate(items):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8, color=TEXT)
    _style(ax, "SciERC training entities by type", "", "entity mentions")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
