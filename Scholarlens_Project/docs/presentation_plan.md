# Presentation plan (15 minutes max, every member speaks, demo required)

Q&A is graded **per person**, so each member must be able to explain their own module and give a one-sentence overview of the others.

| Time | Speaker | Slides / content |
|---|---|---|
| 0:00-1:30 | Member 4 | Title, team, the problem (too many papers, too little time), what ScholarLens does |
| 1:30-3:00 | Member 4 | Architecture diagram + workflow diagram, datasets at a glance (sizes, >= 50k sentences) |
| 3:00-5:00 | Member 1 | Preprocessing (before/after example), summarization: BART vs T5 vs Lead-3/TextRank table, one good and one bad summary |
| 5:00-7:00 | Member 2 | Retrieval (MiniLM vs TF-IDF), QA: DistilBERT vs RoBERTa table, the no-answer threshold, paper-QA domain-shift result |
| 7:00-9:00 | Member 3 | NER: SciBERT vs BERT vs dictionary, per-type F1, a tagged example and a typical error |
| 9:00-12:30 | Member 4 drives, others narrate | **Live demo**: upload a PDF -> summary -> entities -> two questions (one answerable, one the paper does not answer) |
| 12:30-14:00 | Member 1 | Limitations, lessons learned, future work |
| 14:00-15:00 | all | Thank you, then Q&A |

## Demo safety
- Relaunch the Gradio link **within an hour** of presenting (notebook 04) and open it in a browser tab.
- Have a **backup screen recording** of the full demo in case Wi-Fi or Colab fails.
- Pick the demo paper beforehand and test your questions; keep them on a sticky note.

## Q&A practice questions
1. Why did model X beat model Y? (vocabulary, size, pre-training objective)
2. How do you know the results are not inflated? (separate tune/test splits, threshold chosen on tune, fixed seed)
3. What happens when a paper is longer than 1,024 tokens? (truncation vs chunked mode)
4. How does the QA model decide there is no answer?
5. Why is ROUGE not enough on its own? (BERTScore, human reading of samples)
6. What would you do with more time or compute?
7. How did you preprocess the data and why?
