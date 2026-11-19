# Hand-written paper QA test set

SQuAD 2.0 is Wikipedia text, but ScholarLens answers questions about research papers.
To measure how well the pipeline transfers, the team writes a small test set of real paper questions.

**Target: 5 papers x 8-10 questions = 40-50 questions** (about 10-12 per team member).

1. Pick 5 open-access papers (for example from arXiv) that are NOT in the training data you used.
2. Save each paper's text as `eval/papers/<paper_id>.txt` (copy from the PDF, or use the demo's PDF extractor).
3. Write questions in `eval/paper_qa.jsonl`, one JSON object per line:
   `{"paper_id": "<paper_id>", "question": "...", "answers": ["exact span from the paper", "acceptable variant"]}`
   - The answer must be copied **exactly** from the paper text (extractive QA).
   - Include 2-3 questions per paper that the paper does **not** answer, with `"answers": []`.
   - Mix question types: dataset, method, metric, number/result, task, limitation.
4. Write questions **before** looking at model outputs, so the test is fair.
5. Run `python -m src.evaluation.paper_qa` (and once per QA model with `--qa-model`).

See `paper_qa.example.jsonl` and `papers/example_paper.txt` for the format.
