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
    return {
        "bleu1": sentence_bleu(predicted, [reference], smooth_method="exp", tokenize="13a").precisions[0] / 100,
        "bleu4": sentence_bleu(predicted, [reference], smooth_method="exp", tokenize="13a").score / 100,
        "rougeL": scorer.score(reference, predicted)["rougeL"].fmeasure,
        "chrf": sentence_chrf(predicted, [reference]).score / 100,
    }


@torch.inference_mode()
def conditional_ppl(frame: pd.DataFrame, model_name: str, adapter: Path | None, max_length: int) -> list[float]:
    """PPL of answer tokens conditional on the question/context prompt."""
    from peft import PeftModel
    tokenizer = tokenizer_for(model_name)
    model = base_model(model_name)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = next(model.parameters()).device
    values = []
    for row in frame.itertuples(index=False):
        prefix = prompt(row.question, row.context)
        pids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        aids = tokenizer(str(row.predicted_answer) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = (pids + aids)[-max_length:]
        retained_prefix = max(0, len(pids) - max(0, len(pids) + len(aids) - max_length))
        labels = [-100] * retained_prefix + ids[retained_prefix:]
        if not any(label != -100 for label in labels):
            values.append(float("nan")); continue
        tensor = torch.tensor([ids], device=device)
        output = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), labels=torch.tensor([labels], device=device))
        values.append(float(math.exp(min(output.loss.item(), 20))))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return values


def optional_semantic(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cfg["evaluation"].get("compute_bertscore"):
        from bert_score import score
        _, _, f1 = score(frame.predicted_answer.tolist(), frame.long_answer.tolist(), lang="en", verbose=True)
        frame["bertscore_f1"] = f1.cpu().numpy()
    if cfg["evaluation"].get("compute_embedding_cosine"):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(cfg["embedding_model"])
        ref = encoder.encode(frame.long_answer.tolist(), normalize_embeddings=True, show_progress_bar=True)
        pred = encoder.encode(frame.predicted_answer.tolist(), normalize_embeddings=True, show_progress_bar=True)
        frame["embedding_cosine"] = (ref * pred).sum(axis=1)
    return frame
