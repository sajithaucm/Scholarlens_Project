"""Shared helpers: config loading, seeding, paths and result files.

Keeping these in one place means every module saves results in the same
format, which is what src/compare.py expects when it builds the tables.
"""

import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read configs/config.yaml into a dictionary.

    Environment variables SCHOLARLENS_OUTPUT_DIR / SCHOLARLENS_RESULTS_DIR
    override the output paths without editing the file (used on Colab to save
    checkpoints to Google Drive so they survive a disconnect).
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, env in (("output_dir", "SCHOLARLENS_OUTPUT_DIR"), ("results_dir", "SCHOLARLENS_RESULTS_DIR")):
        if os.environ.get(env):
            cfg["paths"][key] = os.environ[env]
    return cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve(path_str: str) -> Path:
    """Turn a config path into an absolute path (relative paths are from the repo root)."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def safe_name(model_name: str) -> str:
    """'facebook/bart-base' -> 'facebook__bart-base' (safe for folder names)."""
    return model_name.replace("/", "__")


def training_settings(cfg: dict, task: str, model_name: str) -> Dict[str, Any]:
    """Merge global training defaults <- task overrides <- model overrides."""
    settings = dict(cfg.get("training", {}))
    settings.update(cfg[task].get("training", {}) or {})
    settings.update((cfg[task].get("model_overrides", {}) or {}).get(model_name, {}) or {})
    return settings


def output_dir(cfg: dict, task: str, model_name: str) -> Path:
    """Where checkpoints for one task/model go (outputs/<task>/<model>)."""
    return resolve(cfg["paths"]["output_dir"]) / task / safe_name(model_name)


def save_results(cfg: dict, task: str, model_name: str, metrics: Dict[str, Any],
                 extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write results/<task>_<model>.json with metrics plus run metadata."""
    results_dir = resolve(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "model": model_name,
        "metrics": metrics,
        "seed": cfg["seed"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": machine_info(),
    }
    if extra:
        payload.update(extra)
    path = results_dir / f"{task}_{safe_name(model_name)}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results -> {path}")
    return path


def machine_info() -> Dict[str, str]:
    """Record the hardware a result came from (reviewers ask about this)."""
    info = {"python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        info["device"] = "unknown"
    return info


def use_fp16(settings: Dict[str, Any], model_name: str) -> bool:
    """fp16 only on GPU, and never for T5 (its activations overflow in fp16 -> NaN loss)."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    return bool(settings.get("fp16", False)) and has_gpu and "t5" not in model_name.lower()


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def warmup_kwargs(ratio: float) -> Dict[str, float]:
    """Linear-warmup setting that works on both transformers 4.x and 5.x.

    transformers 4.x has `warmup_ratio`; 5.x removed it and accepts a float
    in [0, 1) for `warmup_steps` instead.
    """
    import dataclasses
    from transformers import TrainingArguments
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    return {"warmup_ratio": ratio} if "warmup_ratio" in fields else {"warmup_steps": ratio}
