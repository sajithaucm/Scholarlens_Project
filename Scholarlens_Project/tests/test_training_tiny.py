"""End-to-end runs of every training script with tiny models on synthetic data.

Scores here are meaningless (random tiny models); the point is that every code
path - tokenization, training, checkpointing, prediction, post-processing,
metrics, result files, comparison tables and the pipeline - runs without errors.
"""

import json
from pathlib import Path

from src.utils import resolve, safe_name


def _results(cfg, task, model):
    path = resolve(cfg["paths"]["results_dir"]) / f"{task}_{safe_name(model)}.json"
    assert path.exists(), path
    return json.loads(path.read_text())


def test_summarization_training(cfg, tiny_models, summ_data):
    from src.summarization.train import train
    for name in ("tiny-bart", "tiny-t5-like"):
        m = train(tiny_models[name], summ_data, cfg, use_bertscore=False)
        assert {"rouge1", "rouge2", "rougeL", "rougeLsum"} <= set(m)
        r = _results(cfg, "summarization", tiny_models[name])
        assert Path(r["checkpoint"]).joinpath("config.json").exists()


def test_t5_prefix_and_fp16_rules():
    from src.summarization.train import input_prefix
    from src.utils import use_fp16
    c = {"summarization": {"t5_prefix": "summarize: "}}
    assert input_prefix(c, "t5-small") == "summarize: " and input_prefix(c, "facebook/bart-base") == ""
    assert use_fp16({"fp16": True}, "t5-small") is False


def test_qa_training(cfg, tiny_models, qa_data):
    from src.qa.train import prepare_train_features, train
    from transformers import AutoTokenizer

    # Label check: the labelled span must decode back to the gold answer.
    tok = AutoTokenizer.from_pretrained(tiny_models["bert"])
    batch = qa_data["train"][:4]
    feats = prepare_train_features(batch, tok, cfg)
    enc = tok(batch["question"], batch["context"], truncation="only_second", max_length=cfg["qa"]["max_seq_len"],
              stride=cfg["qa"]["doc_stride"], return_overflowing_tokens=True, padding="max_length")
    found = False
    for i, (s, e) in enumerate(zip(feats["start_positions"], feats["end_positions"])):
        gold = batch["answers"][enc["overflow_to_sample_mapping"][i]]["text"]
        if s > 0:
            assert tok.decode(feats["input_ids"][i][s:e + 1]).strip() == gold[0]
            found = True
    assert found

    assert "[UNK]" not in tok.decode(feats["input_ids"][0])

    m = train(tiny_models["bert"], qa_data, cfg)
    assert {"exact_match", "f1", "has_ans_f1", "no_ans_accuracy", "no_answer_threshold"} <= set(m)
    assert m["n_total"] == len(qa_data["test"])


def test_ner_training(cfg, tiny_models, ner_data):
    from src.ner.train import train
    m = train(tiny_models["bert"], ner_data, cfg)
    assert {"precision", "recall", "f1", "per_type_f1"} <= set(m)


def test_compare_builds_tables(cfg, tiny_models, summ_data, qa_data, ner_data, monkeypatch):
    import src.compare as compare
    monkeypatch.setattr(compare, "load_config", lambda: cfg)
    compare.main()
    out = resolve(cfg["paths"]["results_dir"])
    assert (out / "RESULTS.md").exists()
    assert (out / "comparison_qa.csv").exists() and (out / "fig_qa_metrics.png").exists()


def test_pipeline_end_to_end(cfg, tiny_models):
    import copy
    from src.pipeline import ScholarLens
    c = copy.deepcopy(cfg)
    out = resolve(c["paths"]["output_dir"])
    c["demo"] = {
        "summarization_model": str(out / "summarization" / safe_name(tiny_models["tiny-bart"]) / "best"),
        "qa_model": str(out / "qa" / safe_name(tiny_models["bert"]) / "best"),
        "ner_model": str(out / "ner" / safe_name(tiny_models["bert"]) / "best"),
        "retriever_model": "tfidf", "summary_strategy": "chunked", "max_ner_sentences": 20,
    }
    lens = ScholarLens(c, device="cpu")
    assert lens.warnings == [], lens.warnings
    text = " ".join(f"We propose a transformer model number {i} for summarization of papers." for i in range(200))
    assert lens.load_paper(text) > 1
    assert isinstance(lens.summarize("chunked"), str)
    assert isinstance(lens.summarize("truncate"), str)
    assert isinstance(lens.extract_entities(), dict)
    res = lens.answer("Which model is proposed?")
    assert len(res["retrieved"]) == c["retrieval"]["top_k_passages"]
    assert isinstance(lens.tag_text("We use bert on arxiv"), list)

    # Demo callbacks (UI functions) on the same pipeline, without launching a server.
    import app.app as demo_app
    demo_app.LENS = lens
    status, _ = demo_app.load(None, (Path(demo_app.__file__).parent / "example_paper.txt").read_text())
    assert status.startswith("Loaded")
    assert isinstance(demo_app.summarize("truncate"), str)
    md, spans = demo_app.entities()
    assert isinstance(md, str) and isinstance(spans, list)
    ans, ctx = demo_app.ask("Which dataset is used?")
    assert ans.startswith("###") and "similarity" in ctx
    demo_app.build_ui()


def test_qa_postprocessing_recovers_gold(cfg, tiny_models, qa_data):
    """If the logits point at the labelled span, post-processing must return the gold text,
    and must return '' for unanswerable questions (logits point at CLS)."""
    import numpy as np
    from transformers import AutoTokenizer
    from src.evaluation.metrics import qa_metrics
    from src.qa.train import apply_threshold, best_spans, prepare_eval_features, prepare_train_features, references

    tok = AutoTokenizer.from_pretrained(tiny_models["bert"])
    ex = qa_data["test"]
    lab = ex.map(lambda b: prepare_train_features(b, tok, cfg), batched=True, remove_columns=ex.column_names)
    ev = ex.map(lambda b: prepare_eval_features(b, tok, cfg), batched=True, remove_columns=ex.column_names)
    assert len(lab) == len(ev)
    n, L = len(ev), cfg["qa"]["max_seq_len"]
    start, end = np.full((n, L), -10.0), np.full((n, L), -10.0)
    for i in range(n):
        start[i, lab[i]["start_positions"]] = 10.0
        end[i, lab[i]["end_positions"]] = 10.0
    preds = apply_threshold(best_spans(ex, ev, start, end, cfg), 0.0)
    m = qa_metrics(preds, references(ex))
    assert m["exact_match"] == 100.0, m
