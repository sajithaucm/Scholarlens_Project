"""Build the comparison tables and figures for the report from results/*.json.

Every training/baseline script writes results/<task>_<model>.json. This script
collects them into:
  results/comparison_<task>.csv / .md     one row per model (baselines first)
  results/fig_<task>_metrics.png          grouped bar chart of the headline metrics
  results/fig_<task>_loss.png             training/validation loss curves
  results/RESULTS.md                      all tables in one file (paste into report/slides)

Usage:
    python -m src.compare
"""

import json
from pathlib import Path

import pandas as pd

from src.utils import load_config, resolve

# Headline metrics per task, in display order.
HEADLINE = {
    "summarization": ["rouge1", "rouge2", "rougeLsum", "bertscore_f1"],
    "qa": ["exact_match", "f1", "has_ans_f1", "no_ans_accuracy"],
    "ner": ["precision", "recall", "f1", "macro_f1"],
    "retrieval": ["recall_at_1", "recall_at_3", "mrr"],
}
EXTRA = ["train_minutes", "avg_pred_words", "no_answer_threshold"]

# Neutral gray for baselines; fixed categorical order for models (blue, orange, aqua, violet).
BASELINE_COLOR = "#a3a29b"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.stem.endswith("_history"):
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if "task" not in r or "metrics" not in r:
            continue
        row = {"task": r["task"], "model": r["model"]}
        row.update({k: v for k, v in r["metrics"].items() if not isinstance(v, dict)})
        rows.append(row)
    return pd.DataFrame(rows)


def task_table(df: pd.DataFrame, task: str) -> pd.DataFrame:
    t = df[df.task == task].copy()
    if t.empty:
        return t
    t["is_baseline"] = t.model.str.startswith("baseline-") | (t.model == "tfidf")
    t = t.sort_values(["is_baseline", "model"], ascending=[False, True])
    cols = ["model"] + [c for c in HEADLINE[task] + EXTRA if c in t.columns and t[c].notna().any()]
    return t[cols + ["is_baseline"]].reset_index(drop=True)


def plot_metrics(table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [m for m in HEADLINE[task] if m in table.columns]
    if not metrics:
        return
    n = len(table)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(metrics), 3.6), dpi=200)
    model_i = base_i = 0
    for i, row in table.iterrows():
        if row.is_baseline:            # baselines: gray, told apart by texture
            color, hatch = BASELINE_COLOR, ["///", "...", "xx"][base_i % 3]
            base_i += 1
        else:
            color, hatch = MODEL_COLORS[model_i % len(MODEL_COLORS)], None
            model_i += 1
        xs = np.arange(len(metrics)) + (i - (n - 1) / 2) * width
        vals = [row[m] if pd.notna(row[m]) else 0 for m in metrics]
        ax.bar(xs, vals, width * 0.9, color=color, hatch=hatch, edgecolor="white", linewidth=0.8,
               label=row.model)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, color=TEXT)
    ax.set_ylabel("score (0-100)", color=MUTED)
    ax.set_title(f"{task}: test-set comparison", color=TEXT, loc="left", fontsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_losses(results_dir: Path, task: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = sorted(results_dir.glob(f"{task}_*_history.json"))
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=200)
    for i, hp in enumerate(histories):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        name = hp.stem[len(task) + 1:-len("_history")].replace("__", "/")
        log = json.load(open(hp, encoding="utf-8"))
        train = [(e["step"], e["loss"]) for e in log if "loss" in e]
        val = [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]
        if train:
            ax.plot(*zip(*train), color=color, linewidth=2, label=f"{name} train")
        if val:
            ax.plot(*zip(*val), color=color, linewidth=2, linestyle="--", marker="o",
                    markersize=5, label=f"{name} validation")
    ax.set_xlabel("training step", color=MUTED)
    ax.set_ylabel("loss", color=MUTED)
    ax.set_title(f"{task}: training curves", color=TEXT, loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    cfg = load_config()
    results_dir = resolve(cfg["paths"]["results_dir"])
    df = load_results(results_dir)
    if df.empty:
        print(f"No results found in {results_dir}. Run the training/baseline scripts first.")
        return
    sections = ["# ScholarLens results\n", "All scores are on the held-out test split (0-100). "
                "Baselines are listed first. Generated by `python -m src.compare`.\n"]
    for task in HEADLINE:
        table = task_table(df, task)
        if table.empty:
            continue
        shown = table.drop(columns="is_baseline")
        shown.to_csv(results_dir / f"comparison_{task}.csv", index=False)
        md = shown.to_markdown(index=False)
        (results_dir / f"comparison_{task}.md").write_text(md + "\n", encoding="utf-8")
        sections += [f"\n## {task}\n", md + "\n"]
        plot_metrics(table, task, results_dir / f"fig_{task}_metrics.png")
        plot_losses(results_dir, task, results_dir / f"fig_{task}_loss.png")
        print(f"\n{task}\n{md}")
    (results_dir / "RESULTS.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"\nWrote tables and figures to {results_dir}")


if __name__ == "__main__":
    main()
