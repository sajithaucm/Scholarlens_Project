"""ScholarLens demo (Gradio).

Run from the repo root:
    python -m app.app                 # local: http://127.0.0.1:7860
    python -m app.app --share         # Colab: prints a public *.gradio.live link

Upload a PDF or paste paper text, then:
  1. Summary tab   - abstract-style summary from the fine-tuned summarizer
  2. Entities tab  - scientific entities (Task, Method, Metric, Material, ...)
  3. Ask tab       - question answering over the paper (retrieval + extractive QA)
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import ScholarLens  # noqa: E402

EXAMPLE = (Path(__file__).resolve().parent / "example_paper.txt")
LENS: ScholarLens = None


def pdf_to_text(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load(pdf_file, text):
    """Read the uploaded PDF (or pasted text) and index it."""
    if pdf_file is not None:
        text = pdf_to_text(pdf_file if isinstance(pdf_file, str) else pdf_file.name)
    if not text or len(text.split()) < 50:
        return "Please upload a PDF or paste at least 50 words of paper text.", text
    n = LENS.load_paper(text)
    status = f"Loaded {len(LENS.text.split()):,} words -> {n} passages indexed."
    if LENS.warnings:
        status += "\n\nNote: " + " ".join(dict.fromkeys(LENS.warnings))
    return status, text


def summarize(strategy):
    if not LENS.text:
        return "Load a paper first."
    return LENS.summarize(strategy)


def entities():
    if not LENS.text:
        return "Load a paper first.", []
    ents = LENS.extract_entities()
    md = "\n".join(f"**{typ}**: " + ", ".join(vals) for typ, vals in ents.items()) or "No entities found."
    first = LENS.passages[0] if LENS.passages else ""
    return md, LENS.tag_text(first)


def ask(question):
    if not LENS.text:
        return "Load a paper first.", ""
    if not question.strip():
        return "Type a question.", ""
    res = LENS.answer(question)
    answer = res["answer"] or "_The model thinks the paper does not answer this question._"
    ctx = "\n\n".join(f"**[{i + 1}] similarity {r['similarity']}**\n{r['passage']}"
                      for i, r in enumerate(res["retrieved"]))
    return f"### {answer}\n\nconfidence margin: {res['score']}", ctx


def build_ui():
    with gr.Blocks(title="ScholarLens") as demo:
        gr.Markdown("# ScholarLens\nSummaries, scientific entities and question answering for research papers.")
        with gr.Row():
            pdf = gr.File(label="Upload paper (PDF)", file_types=[".pdf"])
            text = gr.Textbox(label="...or paste paper text", lines=8,
                              value=EXAMPLE.read_text(encoding="utf-8") if EXAMPLE.exists() else "")
        load_btn = gr.Button("Load paper", variant="primary")
        status = gr.Markdown()
        load_btn.click(load, [pdf, text], [status, text])

        with gr.Tab("Summary"):
            strategy = gr.Radio(["chunked", "truncate"], value="chunked", label="Strategy",
                                info="truncate = first 1024 tokens (as in training); chunked = summarize sections, then combine")
            sum_btn = gr.Button("Summarize")
            summary = gr.Textbox(label="Summary", lines=8)
            sum_btn.click(summarize, strategy, summary)

        with gr.Tab("Entities"):
            ent_btn = gr.Button("Extract entities")
            ent_md = gr.Markdown()
            highlight = gr.HighlightedText(label="Entities in the first passage", combine_adjacent=True)
            ent_btn.click(entities, None, [ent_md, highlight])

        with gr.Tab("Ask"):
            q = gr.Textbox(label="Question", placeholder="Which dataset do the authors use?")
            ask_btn = gr.Button("Answer")
            ans = gr.Markdown()
            with gr.Accordion("Retrieved passages", open=False):
                ctx = gr.Markdown()
            ask_btn.click(ask, q, [ans, ctx])
            q.submit(ask, q, [ans, ctx])
    return demo


def main():
    global LENS
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="public link (use on Colab)")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    LENS = ScholarLens.from_config()
    for w in LENS.warnings:
        print("WARNING:", w)
    build_ui().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
