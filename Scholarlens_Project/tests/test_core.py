"""Fast tests for preprocessing, data conversion, metrics and baselines (no models)."""

import pytest

from src.baselines import build_gazetteer, gazetteer_tag, lead_k, textrank, tfidf_sentence_answer
from src.data_loading import NER_LABELS, brat_to_sentences, scierc_docs_to_sentences, spans_to_bio
from src.evaluation.metrics import ner_metrics, qa_metrics, retrieval_metrics, summarization_metrics, token_f1
from src.pipeline import bio_to_spans
from src.preprocessing import clean_text, is_valid_pair, split_into_passages, split_sentences


# ----------------------------------------------------------------- preprocessing
def test_clean_text_removes_noise():
    raw = "We use BERT [12, 13] (Devlin et al., 2019) as shown in Fig. 2 , see https://x.org @xmath0 here."
    out = clean_text(raw)
    for junk in ("[12", "Devlin", "Fig.", "https", "@xmath"):
        assert junk not in out
    assert out.startswith("We use BERT") and " ," not in out


def test_split_sentences_protects_abbreviations():
    s = split_sentences("Smith et al. propose a model. It works, e.g. on SQuAD. Results improve.")
    assert s == ["Smith et al. propose a model.", "It works, e.g. on SQuAD.", "Results improve."]


def test_passages_overlap_and_cover_text():
    text = " ".join(f"Sentence number {i} has some words in it." for i in range(60))
    ps = split_into_passages(text, max_words=40, overlap_sentences=1)
    assert len(ps) > 1 and all(len(p.split()) <= 50 for p in ps)
    assert ps[0].split(".")[-2].strip() in ps[1]          # last sentence repeated as overlap


def test_is_valid_pair():
    assert is_valid_pair("w " * 200, "w " * 30)
    assert not is_valid_pair("w " * 50, "w " * 30)        # source too short
    assert not is_valid_pair("w " * 100, "w " * 90)       # summary almost as long as source


# ----------------------------------------------------------------- NER conversion
def test_spans_to_bio_and_overlap():
    tags, skipped = spans_to_bio(5, [(0, 1, "Method"), (1, 1, "Task"), (3, 3, "Metric")])
    assert tags == ["B-Method", "I-Method", "O", "B-Metric", "O"] and skipped == 1


def test_scierc_document_level_offsets():
    doc = {"doc_key": "d1", "sentences": [["We", "use", "BERT", "."], ["F1", "improves", "."]],
           "ner": [[[2, 2, "Method"]], [[4, 4, "Metric"]]]}   # 4 = first token of sentence 2
    rows, skipped = scierc_docs_to_sentences([doc])
    assert rows[0]["ner_tags"] == ["O", "O", "B-Method", "O"]
    assert rows[1]["ner_tags"] == ["B-Metric", "O", "O"] and skipped == 0


def test_scierc_sentence_level_offsets():
    doc = {"sentences": [["A", "b"], ["SVM", "works"]], "ner": [[], [[0, 0, "Method"]]]}
    rows, _ = scierc_docs_to_sentences([doc])
    assert rows[1]["ner_tags"] == ["B-Method", "O"]


def test_brat_fallback():
    text = "We train a support vector machine. Accuracy is high."
    ann = "T1\tMethod 11 33\tsupport vector machine\nT2\tMetric 35 43\tAccuracy\n"
    rows, _ = brat_to_sentences(text, ann, "doc")
    assert rows[0]["tokens"][3:6] == ["support", "vector", "machine"]
    assert rows[0]["ner_tags"][3:6] == ["B-Method", "I-Method", "I-Method"]
    assert rows[1]["ner_tags"][0] == "B-Metric"


def test_label_set():
    assert NER_LABELS[0] == "O" and len(NER_LABELS) == 13


# ----------------------------------------------------------------- metrics
def test_qa_metrics_squad_rules():
    refs = {"a": ["the Transformer"], "b": [], "c": ["BERT base"]}
    preds = {"a": "Transformer", "b": "", "c": "BERT"}
    m = qa_metrics(preds, refs)
    assert m["exact_match"] == pytest.approx(66.67, abs=0.01)
    assert m["no_ans_accuracy"] == 100.0
    assert token_f1("BERT", "BERT base") == pytest.approx(2 / 3)


def test_ner_metrics_entity_level():
    true = [["B-Method", "I-Method", "O", "B-Task"]]
    pred = [["B-Method", "I-Method", "O", "O"]]
    m = ner_metrics(true, pred)
    assert m["precision"] == 100.0 and m["recall"] == 50.0 and m["per_type_f1"]["Task"] == 0.0


def test_summarization_metrics_rouge():
    m = summarization_metrics(["the cat sat on the mat."], ["the cat sat on the mat."], use_bertscore=False)
    assert m["rouge1"] == 100.0 and m["rougeLsum"] == 100.0


def test_retrieval_metrics():
    m = retrieval_metrics([[False, True, False], [True, False, False], [False, False, False]], k=3)
    assert m["recall_at_1"] == pytest.approx(33.33, abs=0.01)
    assert m["recall_at_3"] == pytest.approx(66.67, abs=0.01)
    assert m["mrr"] == pytest.approx(50.0, abs=0.01)


# ----------------------------------------------------------------- baselines
def test_summarization_baselines():
    art = " ".join(f"Sentence {i} talks about transformers and attention models." for i in range(20))
    assert lead_k(art, 3).count(".") == 3
    tr = textrank(art, k=4)
    assert 0 < len(split_sentences(tr)) <= 4


def test_qa_baseline_picks_relevant_sentence():
    ctx = "The sky is blue. BERT was trained on Wikipedia and books. Cats sleep a lot."
    assert "Wikipedia" in tfidf_sentence_answer("What was BERT trained on?", ctx)


def test_gazetteer_longest_match():
    names = NER_LABELS
    ds = {"tokens": [["support", "vector", "machine", "is", "good"]],
          "ner_tags": [[names.index("B-Method"), names.index("I-Method"), names.index("I-Method"), 0, 0]]}
    gaz = build_gazetteer(ds, names)
    assert gazetteer_tag(["a", "Support", "vector", "machine"], gaz) == ["O", "B-Method", "I-Method", "I-Method"]


def test_bio_to_spans():
    spans = bio_to_spans(["We", "use", "BERT", "base", "on", "SQuAD"],
                         ["O", "O", "B-Method", "I-Method", "O", "B-Material"])
    assert spans == [("Method", "BERT base"), ("Material", "SQuAD")]


def test_tfidf_retriever():
    from src.qa.retriever import Retriever
    r = Retriever("tfidf").index(["Cats are animals.", "BERT is a transformer language model.", "Rain."])
    assert "BERT" in r.search("Which transformer model?", top_k=1)[0][0]


def test_eda_helpers(tmp_path):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
    from src import eda
    s = DatasetDict({"train": Dataset.from_list([{"article": "One sentence here. Another one. " * 20,
                                                   "abstract": "Short abstract text here."}] * 5)})
    st = eda.summarization_stats(s)
    assert st["train"]["est_total_sentences"] > 0
    eda.plot_length_hist([1, 2, 3, 4], [5, 6, 7], "a", "b", "t", tmp_path / "h.png")
    feats = Features({"id": Value("string"), "tokens": Sequence(Value("string")),
                      "ner_tags": Sequence(ClassLabel(names=NER_LABELS))})
    n = DatasetDict({"train": Dataset.from_list([{"id": "0", "tokens": ["BERT", "x"],
                                                  "ner_tags": ["B-Method", "O"]}], features=feats)})
    ns = eda.ner_stats(n)
    assert ns["train"]["n_Method"] == 1
    eda.plot_entity_counts(ns, tmp_path / "e.png")
    assert (tmp_path / "e.png").exists()
