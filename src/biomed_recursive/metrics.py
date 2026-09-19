from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu, sentence_chrf
from transformers import pipeline

from .common import prompt
from .modeling import base_model, tokenizer_for


def words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def repetition_3gram(text: str) -> float:
    grams = list(zip(*(words(text)[i:] for i in range(3))))
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def distinct(texts: list[str], n: int) -> float:
    grams = [tuple(tokens[i:i+n]) for text in texts for tokens in [words(text)] for i in range(max(0, len(tokens)-n+1))]
    return 0.0 if not grams else len(set(grams)) / len(grams)


def normalise_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class EntityExtractor:
    def __init__(self, model_name: str):
        self.pipe = pipeline("token-classification", model=model_name, aggregation_strategy="simple",
                             device=0 if torch.cuda.is_available() else -1)

    def extract(self, text: str) -> dict[str, set[str]]:
        result = {"disease": set(), "chemical": set()}
        # PubMedQA contexts can exceed the 512-position BERT limit. Process
        # overlapping tokenizer chunks rather than dropping the tail; output
        # sets make duplicate entities from the overlap harmless.
        tokenizer = self.pipe.tokenizer
        token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
        window = min(510, tokenizer.model_max_length - 2)
        stride = 32
        for start in range(0, len(token_ids), max(1, window - stride)):
            chunk = tokenizer.decode(token_ids[start:start + window], skip_special_tokens=True)
            if not chunk.strip():
                continue
            for item in self.pipe(chunk):
                label = item["entity_group"].lower()
                if "disease" in label:
                    result["disease"].add(normalise_entity(item["word"]))
                elif "chemical" in label or "drug" in label:
                    result["chemical"].add(normalise_entity(item["word"]))
        return result


def entity_values(reference: set[str], predicted: set[str], context: set[str], prefix: str) -> dict[str, float]:
    overlap = reference & predicted
    precision = len(overlap) / len(predicted) if predicted else (1.0 if not reference else 0.0)
    recall = len(overlap) / len(reference) if reference else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + (0 if np.isnan(recall) else recall) else 0.0
    mismatch = len(predicted - reference) / len(predicted) if predicted else 0.0
    novel = predicted - reference
    return {
        f"entity_{prefix}_precision": precision, f"entity_{prefix}_recall": recall,
        f"entity_{prefix}_f1": f1, f"entity_{prefix}_retention": recall,
        f"entity_{prefix}_reference_mismatch": mismatch,
        f"entity_{prefix}_context_supported_new": len(novel & context) / len(predicted) if predicted else 0.0,
        f"entity_{prefix}_unsupported": len(novel - context) / len(predicted) if predicted else 0.0,
        f"entity_{prefix}_gold_unique_count": len(reference), f"entity_{prefix}_pred_unique_count": len(predicted),
    }


def lexical_values(reference: str, predicted: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    values = {
        "bleu1": sentence_bleu(predicted, [reference], smooth_method="exp", tokenize="13a").precisions[0] / 100,
        "bleu4": sentence_bleu(predicted, [reference], smooth_method="exp", tokenize="13a").score / 100,
        "rougeL": scorer.score(reference, predicted)["rougeL"].fmeasure,
        "chrf": sentence_chrf(predicted, [reference]).score / 100,
    }
    return values


@torch.inference_mode()
def conditional_ppl(frame: pd.DataFrame, model_name: str, adapter: Path | None,
                    max_length: int, batch_size: int = 1) -> list[float]:
    """Per-answer conditional PPL with batched inference and masked padding."""
    from peft import PeftModel
    if batch_size < 1:
        raise ValueError("Perplexity batch_size must be positive")
    tokenizer = tokenizer_for(model_name)
    model = base_model(model_name)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(frame), batch_size):
        inputs, targets = [], []
        for row in frame.iloc[start:start + batch_size].itertuples(index=False):
            prefix = prompt(row.question, row.context)
            pids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            aids = tokenizer(str(row.predicted_answer) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = (pids + aids)[-max_length:]
            retained_prefix = max(0, len(pids) - max(0, len(pids) + len(aids) - max_length))
            inputs.append({"input_ids": ids})
            targets.append([-100] * retained_prefix + ids[retained_prefix:])
        batch = tokenizer.pad(inputs, padding=True, return_tensors="pt").to(device)
        labels = torch.full_like(batch.input_ids, -100)
        for i, target in enumerate(targets):
            labels[i, -len(target):] = torch.tensor(target, device=device)
        logits = model(**batch).logits[:, :-1, :].float()
        shifted = labels[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1),
            reduction="none", ignore_index=-100,
        ).reshape(shifted.shape)
        for loss, valid in zip(losses, shifted != -100):
            mean = loss[valid].mean().item() if valid.any() else float("nan")
            values.append(math.exp(min(mean, 20)))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values


def optional_semantic(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cfg["evaluation"].get("compute_bertscore"):
        from bert_score import score
        _, _, f1 = score(frame.predicted_answer.tolist(), frame.long_answer.tolist(), lang="en", verbose=True, batch_size=cfg["evaluation"].get("bertscore_batch_size", 32))
        frame["bertscore_f1"] = f1.cpu().numpy()
    if cfg["evaluation"].get("compute_embedding_cosine"):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(cfg["embedding_model"])
        ref = encoder.encode(frame.long_answer.tolist(), normalize_embeddings=True, show_progress_bar=True, batch_size=cfg["evaluation"].get("embedding_batch_size", 32))
        pred = encoder.encode(frame.predicted_answer.tolist(), normalize_embeddings=True, show_progress_bar=True, batch_size=cfg["evaluation"].get("embedding_batch_size", 32))
        frame["embedding_cosine"] = (ref * pred).sum(axis=1)
    return frame
