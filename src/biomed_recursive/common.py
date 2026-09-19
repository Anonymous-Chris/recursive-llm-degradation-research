from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)


def context_to_text(context: Any) -> str:
    """Normalise PubMedQA's list/dict/string context representation."""
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        for key in ("contexts", "abstract", "text"):
            if key in context:
                return context_to_text(context[key])
        return " ".join(context_to_text(v) for v in context.values())
    if isinstance(context, (list, tuple)):
        return " ".join(context_to_text(item) for item in context)
    return str(context)


def prompt(question: str, context: str) -> str:
    return (
        "You are a biomedical research assistant. Answer the question using only "
        "the supplied context. Give a concise, evidence-grounded long answer.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n"
    )


def normalize_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"pubid", "question", "context", "long_answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    for col in ("pubid", "question", "context", "long_answer"):
        result[col] = result[col].fillna("").astype(str)
    if "final_decision" not in result:
        result["final_decision"] = ""
    return result


def validate_splits(human: pd.DataFrame, gold: pd.DataFrame) -> None:
    for name, frame in (("human", human), ("gold", gold)):
        ids = frame.pubid.astype(str).str.strip()
        if ids.eq("").any() or ids.duplicated().any():
            raise ValueError(f"{name} pubids must be non-empty and unique")
    if set(human.pubid.astype(str)) & set(gold.pubid.astype(str)):
        raise ValueError("Human training and gold evaluation pubids overlap")
