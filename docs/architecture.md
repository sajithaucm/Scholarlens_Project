# ScholarLens architecture

Two diagrams are required by the course ("system architecture and pipeline diagrams").
GitHub renders the Mermaid code below directly. PNG versions for the report and slides
are in `docs/diagrams/` (re-render after edits with
`mmdc -i docs/diagrams/architecture.mmd -o docs/diagrams/architecture.png -s 3 -b white`,
or paste the code into https://mermaid.live and export).

## 1. System architecture (inference / demo)

![System architecture](diagrams/architecture.png)

```mermaid
flowchart LR
    U([User: PDF or pasted text]) --> P[PDF text extraction<br/>pypdf]
    P --> C[Cleaning<br/>LaTeX, citations, URLs,<br/>figure refs removed]
    C --> S[Sentence split +<br/>120-word overlapping passages]

    subgraph SUM[Summarization]
        S --> SM[Fine-tuned BART-base<br/>vs T5-small]
        SM --> SO[Abstract-style summary]
    end

    subgraph NER[Scientific NER]
        S --> NM[Fine-tuned SciBERT<br/>vs BERT-base<br/>BIO tagging]
        NM --> NO[Tasks, Methods, Metrics,<br/>Materials, Other terms]
    end

    subgraph QA[Question answering]
        Q([User question]) --> R
        S --> IDX[Passage index]
        IDX --> R[Retriever<br/>MiniLM embeddings vs TF-IDF]
        R --> TOP[Top-3 passages]
        TOP --> QM[Fine-tuned RoBERTa<br/>vs DistilBERT<br/>span + no-answer]
        QM --> QO[Answer + evidence passage]
    end

    SO --> UI[Gradio demo]
    NO --> UI
    QO --> UI
```

**How to describe it in the report:** a paper enters as a PDF or text, is cleaned once by
`src/preprocessing.py`, and the same cleaned text feeds three branches. Summarization reads the
first 1,024 tokens (or chunk summaries in the demo's chunked mode). NER tags each sentence with
BIO labels. QA first narrows the paper to three passages with a retriever, because the QA models
only read 384 tokens at a time, then extracts a span or abstains.

## 2. Training and evaluation workflow

![Workflow](diagrams/workflow.png)

```mermaid
flowchart TB
    subgraph DATA[1. Data]
        D1[arXiv summarization<br/>10k papers] --> PRE[Preprocessing +<br/>quality filter]
        D2[SQuAD 2.0<br/>40k questions] --> PRE
        D3[SciERC<br/>500 abstracts] --> BIO[Span to BIO<br/>conversion]
        PRE --> SPL[Fixed-seed splits<br/>train / validation-tune / test]
        BIO --> SPL
    end
    subgraph TRAIN[2. Training - Colab T4 GPU]
        SPL --> B[Baselines<br/>Lead-3, TextRank, TF-IDF,<br/>no-answer, dictionary]
        SPL --> FT[Fine-tune 2 models per task<br/>same config, seed 42]
        FT --> SEL[Best checkpoint on<br/>validation / tune split]
        SEL --> THR[QA: no-answer threshold<br/>tuned on tune split]
    end
    subgraph EVAL[3. Evaluation - test split, used once]
        B --> M[ROUGE-1/2/L, BERTScore<br/>EM, F1, HasAns/NoAns<br/>entity P/R/F1, Recall@k, MRR]
        THR --> M
        SEL --> M
        M --> CMP[Comparison tables +<br/>figures + error analysis]
        HW[Hand-written paper<br/>questions] --> CMP
    end
    CMP --> REP[IEEE report + demo + slides]
```
