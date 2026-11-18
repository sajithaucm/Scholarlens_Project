"""Dataset loaders for the three ScholarLens tasks.

Each loader reads sizes and names from configs/config.yaml, cleans the data
with src/preprocessing.py, subsamples reproducibly (fixed seed) and returns a
Hugging Face DatasetDict ready for tokenization.

Split policy (important for honest evaluation):
  * Summarization: official arXiv train / validation / test splits.
  * QA: SQuAD 2.0 has no public test set, so the official dev set is our TEST
    set and is used once at the end. Model selection uses a "tune" split held
    out from the training data.
  * NER: official SciERC train / dev / test splits.

Quick smoke test of any task:
    python -m src.data_loading --task summarization --max-samples 200
"""

import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value, load_dataset

from src.preprocessing import clean_text, is_valid_pair, split_sentences
from src.utils import load_config, resolve  # noqa: F401  (load_config re-exported for old imports)

NER_ENTITY_TYPES = ["Task", "Method", "Metric", "Material", "OtherScientificTerm", "Generic"]
NER_LABELS = ["O"] + [f"{p}-{t}" for t in NER_ENTITY_TYPES for p in ("B", "I")]


def _subsample(ds, n, seed: int):
    """Shuffle with a fixed seed and keep at most n rows (n=None keeps all)."""
    if n is None:
        return ds
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))


def _cap(n, max_samples):
    """Apply the --max-samples cap used for quick tests."""
    if max_samples is None:
        return n
    return max_samples if n is None else min(n, max_samples)


# ---------------------------------------------------------------- Summarization
def load_summarization(cfg: dict, max_samples: int = None) -> DatasetDict:
    """arXiv papers (`article`) paired with their abstracts (`abstract`)."""
    c, seed = cfg["summarization"], cfg["seed"]
    raw = load_dataset(c["dataset"])  # splits: train / validation / test

    def clean(example):
        return {
            "article": clean_text(example["article"]),
            "abstract": clean_text(example["abstract"]),
        }

    sizes = {"train": c["train_samples"], "validation": c["val_samples"], "test": c["test_samples"]}
    out = {}
    for split, n in sizes.items():
        n = _cap(n, max_samples)
        # Oversample a little because the quality filter removes some rows.
        ds = _subsample(raw[split], int(n * 1.2), seed).map(clean)
        ds = ds.filter(lambda ex: is_valid_pair(ex["article"], ex["abstract"]))
        out[split] = ds.select(range(min(n, len(ds))))
    return DatasetDict(out)


# --------------------------------------------------------------------------- QA
def load_qa(cfg: dict, max_samples: int = None) -> DatasetDict:
    """SQuAD 2.0 -> train / tune (held out from train) / test (official dev set)."""
    c, seed = cfg["qa"], cfg["seed"]
    raw = load_dataset(c["dataset"])  # splits: train / validation

    def clean(example):
        # Only the question is cleaned: answer start offsets index into the
        # context, so editing the context would corrupt the labels.
        return {"question": " ".join(example["question"].split())}

    n_train = _cap(c["train_samples"], max_samples)
    pool = _subsample(raw["train"], n_train, seed).map(clean)
    split = pool.train_test_split(test_size=c.get("tune_fraction", 0.05), seed=seed)
    test = _subsample(raw["validation"], _cap(c.get("test_samples"), max_samples), seed).map(clean)
    return DatasetDict({"train": split["train"], "tune": split["test"], "test": test})


# -------------------------------------------------------------------------- NER
def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _safe_extract(tar_path: Path, target: Path) -> None:
    """Extract a tar file without letting entries escape the target folder."""
    target = target.resolve()
    with tarfile.open(tar_path) as tar:
        members = []
        for m in tar.getmembers():
            if not (target / m.name).resolve().is_relative_to(target) or m.issym() or m.islnk():
                continue
            members.append(m)
        tar.extractall(target, members=members)


def spans_to_bio(n_tokens: int, spans: List[Tuple[int, int, str]]) -> Tuple[List[str], int]:
    """Convert inclusive token spans [(start, end, label)] to BIO tags.

    Returns (tags, n_skipped). Overlapping spans are rare in SciERC; longer
    spans win and the shorter overlapping span is skipped (and counted so the
    report can mention it).
    """
    tags = ["O"] * n_tokens
    skipped = 0
    for start, end, label in sorted(spans, key=lambda s: s[0] - s[1]):  # longest first
        if start < 0 or end >= n_tokens or start > end or any(t != "O" for t in tags[start:end + 1]):
            skipped += 1
            continue
        tags[start] = f"B-{label}"
        for i in range(start + 1, end + 1):
            tags[i] = f"I-{label}"
    return tags, skipped


def scierc_docs_to_sentences(docs: List[dict]) -> Tuple[List[dict], int]:
    """Turn SciERC (DyGIE-style JSON) documents into one BIO example per sentence.

    In this format `sentences` is a list of token lists and `ner` holds, for
    each sentence, spans [start, end, label] whose indices count tokens from the
    start of the whole document (inclusive end). Some copies use sentence-level
    indices instead, so we detect which one each document uses.
    """
    rows, skipped = [], 0
    for doc in docs:
        sentences, ner = doc["sentences"], doc.get("ner", [[] for _ in doc["sentences"]])
        offsets, total = [], 0
        for sent in sentences:
            offsets.append(total)
            total += len(sent)
        doc_level = all(
            offsets[i] <= s and e < offsets[i] + len(sentences[i])
            for i, spans in enumerate(ner) for s, e, *_ in spans
        )
        for i, (sent, spans) in enumerate(zip(sentences, ner)):
            shift = offsets[i] if doc_level else 0
            local = [(s - shift, e - shift, lab) for s, e, lab, *_ in spans]
            tags, n_skip = spans_to_bio(len(sent), local)
            skipped += n_skip
            rows.append({"id": f"{doc.get('doc_key', 'doc')}_{i}", "tokens": list(sent), "ner_tags": tags})
    return rows, skipped


_TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


def brat_to_sentences(text: str, ann: str, doc_id: str) -> Tuple[List[dict], int]:
    """Fallback for the raw SciERC release (brat .txt + .ann files, character offsets)."""
    spans = []
    for line in ann.splitlines():
        if not line.startswith("T"):
            continue
        _, info, *_ = line.split("\t")
        label, *offs = info.replace(";", " ").split()
        if label in NER_ENTITY_TYPES:
            spans.append((int(offs[0]), int(offs[-1]), label))
    rows, skipped, cursor = [], 0, 0
    for i, sent in enumerate(split_sentences(text)):
        base = text.find(sent, cursor)
        base = cursor if base < 0 else base
        cursor = base + len(sent)
        toks = [(m.group(), base + m.start(), base + m.end()) for m in _TOKEN_RE.finditer(sent)]
        local = []
        for cs, ce, label in spans:
            idx = [k for k, (_, ts, te) in enumerate(toks) if ts < ce and te > cs]
            if idx:
                local.append((idx[0], idx[-1], label))
        tags, n_skip = spans_to_bio(len(toks), local)
        skipped += n_skip
        rows.append({"id": f"{doc_id}_{i}", "tokens": [t for t, _, _ in toks], "ner_tags": tags})
    return rows, skipped


def _ner_features() -> Features:
    return Features({
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=NER_LABELS)),
    })


def _rows_to_dataset(rows: List[dict]) -> Dataset:
    return Dataset.from_list(rows, features=_ner_features())


def _find_split_files(root: Path) -> Dict[str, Path]:
    found = {}
    for split, name in (("train", "train.json"), ("validation", "dev.json"), ("test", "test.json")):
        candidates = sorted(root.rglob(name), key=lambda p: ("json" not in p.parts, len(p.parts)))
        if candidates:
            found[split] = candidates[0]
    return found


def load_ner(cfg: dict, max_samples: int = None) -> DatasetDict:
    """SciERC scientific NER as BIO-tagged sentences (train / validation / test)."""
    c, seed = cfg["ner"], cfg["seed"]
    local = resolve(c["local_dir"])
    local.mkdir(parents=True, exist_ok=True)

    files = _find_split_files(local)
    if len(files) < 3:
        try:
            _safe_extract(_download(c["processed_url"], local / "sciERC_processed.tar.gz"), local)
            files = _find_split_files(local)
        except Exception as err:
            print(f"Processed SciERC download failed ({err}); trying the raw release.")

    out, skipped = {}, 0
    if len(files) == 3:
        for split, path in files.items():
            with open(path) as f:
                docs = [json.loads(line) for line in f if line.strip()]
            rows, n_skip = scierc_docs_to_sentences(docs)
            skipped += n_skip
            out[split] = _rows_to_dataset(rows)
        source = f"SciERC processed JSON ({files['train'].parent})"
    else:
        try:
            _safe_extract(_download(c["raw_url"], local / "sciERC_raw.tar.gz"), local)
        except Exception as err:
            raise RuntimeError(
                "Could not get SciERC. Download sciERC_processed.tar.gz from "
                "http://nlp.cs.washington.edu/sciIE/ and extract it into "
                f"{local} (it must contain train.json, dev.json, test.json)."
            ) from err
        docs = []
        for txt in sorted(local.rglob("*.txt")):
            ann = txt.with_suffix(".ann")
            if ann.exists():
                rows, n_skip = brat_to_sentences(txt.read_text(), ann.read_text(), txt.stem)
                skipped += n_skip
                docs.append(rows)
        if not docs:
            raise RuntimeError(f"No .txt/.ann pairs found under {local}")
        # The raw release has no official split: split by DOCUMENT (not sentence)
        # with a fixed seed so sentences of one abstract never leak across splits.
        import random
        random.Random(seed).shuffle(docs)
        n = len(docs)
        cut1, cut2 = int(0.7 * n), int(0.8 * n)
        parts = {"train": docs[:cut1], "validation": docs[cut1:cut2], "test": docs[cut2:]}
        out = {k: _rows_to_dataset([r for d in v for r in d]) for k, v in parts.items()}
        source = "SciERC raw brat files (document-level 70/10/20 split)"

    print(f"NER data: {source}; overlapping spans skipped: {skipped}")
    if max_samples:
        out = {k: _subsample(v, max_samples, seed) for k, v in out.items()}
    return DatasetDict(out)


LOADERS = {"summarization": load_summarization, "qa": load_qa, "ner": load_ner}


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a ScholarLens dataset loader")
    parser.add_argument("--task", choices=LOADERS.keys(), required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    data = LOADERS[args.task](cfg, args.max_samples)
    for split, ds in data.items():
        print(f"{split}: {len(ds)} examples, columns = {ds.column_names}")
    first_split = next(iter(data))
    print("\nSample row:\n", data[first_split][0])


if __name__ == "__main__":
    main()
