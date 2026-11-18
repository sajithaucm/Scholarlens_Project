# Rubric checklist (from Project_Guidance.docx)

Tick each item before Dec 3. "Where" says which file or result proves it, so you can point to it in the report and during Q&A.

## Deadlines
- [ ] Final report submitted — **Dec 3**
- [ ] Presentation + live demo (10-15 min, every member speaks) — **Dec 3**
- [ ] GitHub link submitted on **Brightspace**

## Submission
| Requirement | Where | Status |
|---|---|---|
| Source code + documentation on GitHub | this repo | [ ] |
| README has team information | `README.md` -> Team table (replace every TODO) | [ ] |
| README has a short introduction | `README.md` -> Introduction | [x] |
| Code commented appropriately | docstrings + comments in `src/`, `app/` | [x] |
| Report is unplagiarized | written by the team; AI only for help, as the guidance allows | [ ] |
| Report in IEEE format | `docs/report/main.tex` (IEEEtran, Overleaf) | [ ] |
| Report sections: Introduction, Objectives, Approaches/Methods, Workflow, Datasets, Parameters, Evaluation & Discussion, Conclusion, References | all present in `main.tex` | [ ] filled in |
| Detailed paragraphs, not bullet summaries | every section of the report | [ ] |

## Technical requirements
| Requirement | How ScholarLens meets it | Evidence |
|---|---|---|
| Original problem, real-world use | Assistant for reading research papers | Report Introduction / Objectives |
| Not a single-task toy; several NLP techniques | Summarization + NER + semantic retrieval + QA in one pipeline | `src/pipeline.py`, architecture diagram |
| Advanced transformer you fine-tune | BART, T5, BERT, SciBERT, DistilBERT, RoBERTa, all fine-tuned | `src/*/train.py`, `outputs/` checkpoints |
| Several NLP tasks | Summarization, NER, semantic similarity (retrieval), QA | same |
| Large real dataset (>= 50k sentences) + preprocessing/cleaning | arXiv 10k papers (millions of sentences), SQuAD 2.0 (40k questions), SciERC; cleaning in `src/preprocessing.py` | `results/dataset_stats.md` (est_total_sentences) |
| At least 3 metrics | Summ: ROUGE-1/2/Lsum + BERTScore; QA: EM, F1, HasAns-F1, NoAns-acc; NER: P/R/F1/macro-F1; retrieval: R@1, R@3, MRR | `results/RESULTS.md` |
| Compare at least 2 model variants | BART vs T5, DistilBERT vs RoBERTa, SciBERT vs BERT, MiniLM vs TF-IDF (+ baselines) | `results/comparison_*.md`, `fig_*_metrics.png` |
| System architecture + pipeline diagrams | `docs/architecture.md`, `docs/diagrams/*.png` | in report + slides |
| Demo shown in presentation | `app/app.py` (Gradio), `notebooks/04_results_and_demo.ipynb` | live link + backup video |

## Things that commonly lose points (check these too)
- [ ] No test-set leakage: QA threshold/checkpoint chosen on `tune`, never on `test` (already enforced in code)
- [ ] Every number in the report matches a file in `results/`
- [ ] Error analysis with concrete examples (not only tables)
- [ ] Limitations stated honestly (input truncation, Wikipedia-to-papers domain shift, small SciERC)
- [ ] Hyperparameters and hardware reported (from `results/*.json`)
- [ ] Fresh clone + README instructions work (a teammate tries it on a clean Colab)
- [ ] Every member can explain every part (Q&A is graded per person)
