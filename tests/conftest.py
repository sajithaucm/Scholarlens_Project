"""Test fixtures: tiny randomly initialised models and synthetic datasets.

These let the full training / evaluation code paths run in under a minute on
a laptop CPU without downloading anything, so a broken script is caught before
anyone spends an hour of Colab GPU time on it.
"""

import copy
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORDS = ("model method dataset metric task we propose a new transformer for summarization of "
         "scientific papers and evaluate it with rouge on arxiv results show strong gains over "
         "baselines the approach uses attention encoder decoder training data accuracy score "
         "question answering retrieval entity recognition bert language network learning").split()


def sentence(rng, n=10):
    words = [rng.choice(WORDS) for _ in range(n)]
    return " ".join(words).capitalize() + "."


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    from src.utils import load_config
    c = copy.deepcopy(load_config())
    tmp = tmp_path_factory.mktemp("run")
    c["paths"] = {"data_dir": str(tmp / "data"), "output_dir": str(tmp / "outputs"),
                  "results_dir": str(tmp / "results")}
    c["training"].update({"epochs": 1, "batch_size": 4, "logging_steps": 1, "fp16": False})
    for task in ("summarization", "qa", "ner"):
        c[task]["training"] = {"epochs": 1, "batch_size": 4, "grad_accum_steps": 1}
        c[task].pop("model_overrides", None)
    c["summarization"]["max_input_tokens"] = 128
    c["summarization"]["max_target_tokens"] = 32
    c["summarization"]["generation"] = {"num_beams": 2, "max_new_tokens": 12, "min_new_tokens": 2}
    c["qa"]["max_seq_len"] = 64
    c["qa"]["doc_stride"] = 16
    c["ner"]["max_seq_len"] = 64
    return c


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """Build tiny BERT (encoder) and BART (seq2seq) checkpoints on disk."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (BartConfig, BartForConditionalGeneration, BartTokenizerFast, BertConfig,
                              BertModel, BertTokenizerFast)

    base = tmp_path_factory.mktemp("models")
    rng = random.Random(0)

    # --- tiny BERT
    bert_dir = base / "tiny-bert"
    bert_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS)) + list(".,?!-()'")
    vocab += [f"##{c}" for c in "abcdefghijklmnopqrstuvwxyz0123456789"] + list("abcdefghijklmnopqrstuvwxyz0123456789")
    (bert_dir / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)))
    BertTokenizerFast.from_pretrained(str(bert_dir), do_lower_case=True).save_pretrained(bert_dir)
    bcfg = BertConfig(vocab_size=len(dict.fromkeys(vocab)), hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=2, intermediate_size=64, max_position_embeddings=256)
    BertModel(bcfg).save_pretrained(bert_dir)

    # --- tiny BART (the "t5" name exercises the T5 prefix + fp16-off code paths)
    corpus = base / "corpus.txt"
    corpus.write_text("\n".join(sentence(rng, 12) for _ in range(300)))
    bpe = ByteLevelBPETokenizer()
    bpe.train([str(corpus)], vocab_size=400, min_frequency=1,
              special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    out = {"bert": str(bert_dir)}
    for name in ("tiny-bart", "tiny-t5-like"):
        d = base / name
        d.mkdir()
        bpe.save_model(str(d))
        tok = BartTokenizerFast.from_pretrained(str(d))
        tok.save_pretrained(d)
        mcfg = BartConfig(vocab_size=len(tok), d_model=32, encoder_layers=1, decoder_layers=1,
                          encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64,
                          decoder_ffn_dim=64, max_position_embeddings=256, pad_token_id=tok.pad_token_id,
                          bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                          decoder_start_token_id=tok.eos_token_id, forced_bos_token_id=None)
        BartForConditionalGeneration(mcfg).save_pretrained(d)
        out[name] = str(d)
    return out


@pytest.fixture(scope="session")
def summ_data():
    from datasets import Dataset, DatasetDict
    rng = random.Random(1)

    def rows(n):
        return [{"article": " ".join(sentence(rng) for _ in range(12)),
                 "abstract": " ".join(sentence(rng) for _ in range(3))} for _ in range(n)]
    return DatasetDict({k: Dataset.from_list(rows(n)) for k, n in
                        (("train", 16), ("validation", 4), ("test", 6))})


@pytest.fixture(scope="session")
def qa_data():
    from datasets import Dataset, DatasetDict
    rng = random.Random(2)

    def rows(n, prefix):
        out = []
        for i in range(n):
            ctx = " ".join(sentence(rng) for _ in range(8))   # long enough to need 2+ windows
            ans = "transformer encoder"
            pos = rng.randint(0, len(ctx) // 2)
            pos = ctx.find(" ", pos) + 1
            ctx = ctx[:pos] + ans + " " + ctx[pos:]
            answerable = i % 2 == 0
            out.append({"id": f"{prefix}{i}", "title": "t", "context": ctx,
                        "question": "Which model is used?",
                        "answers": {"text": [ans] if answerable else [],
                                    "answer_start": [pos] if answerable else []}})
        return out
    return DatasetDict({"train": Dataset.from_list(rows(24, "tr")), "tune": Dataset.from_list(rows(8, "tu")),
                        "test": Dataset.from_list(rows(10, "te"))})


@pytest.fixture(scope="session")
def ner_data():
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
    from src.data_loading import NER_LABELS
    rng = random.Random(3)
    feats = Features({"id": Value("string"), "tokens": Sequence(Value("string")),
                      "ner_tags": Sequence(ClassLabel(names=NER_LABELS))})

    def rows(n):
        out = []
        for i in range(n):
            toks = [rng.choice(WORDS) for _ in range(12)] + ["bert", "model", "on", "arxiv"]
            tags = ["O"] * 12 + ["B-Method", "I-Method", "O", "B-Material"]
            out.append({"id": str(i), "tokens": toks, "ner_tags": tags})
        return out
    return DatasetDict({k: Dataset.from_list(rows(n), features=feats)
                        for k, n in (("train", 24), ("validation", 6), ("test", 6))})
