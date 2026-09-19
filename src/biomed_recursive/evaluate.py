from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_config
from .metrics import EntityExtractor, conditional_ppl, distinct, entity_values, lexical_values, optional_semantic, repetition_3gram, words


def decision_from_terminal(text: str) -> str:
    match = __import__("re").search(r"\b(yes|no|maybe)\b[.!]?\s*$", str(text).lower())
    return match.group(1) if match else "unmapped"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all generation predictions in one run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    cfg, run_dir = load_config(args.config), Path(args.run_dir)
    gold = pd.read_csv(Path(cfg["data_root"]) / "gold_eval.csv", dtype={"pubid": str})
    extractor = EntityExtractor(cfg["ner_model"])
    summaries = []
    for generation in range(cfg["generations"] + 1):
        path = run_dir / f"g{generation}_predictions.csv"
        pred = pd.read_csv(path, dtype={"pubid": str})
        if pred.pubid.tolist() != gold.pubid.tolist():
            raise ValueError(f"{path} is not the fixed gold set in the same order.")
        if "long_answer" not in pred: pred["long_answer"] = gold.long_answer
        rows = []
        for row in pred.itertuples(index=False):
            ref, generated, context = extractor.extract(row.long_answer), extractor.extract(row.predicted_answer), extractor.extract(row.context)
            values = {"pubid": row.pubid, "generation": f"G{generation}", "answer_words": len(words(row.predicted_answer)),
                      "answer_tokens": len(words(row.predicted_answer)), "repetition_3gram": repetition_3gram(row.predicted_answer)}
            values.update(lexical_values(row.long_answer, row.predicted_answer))
            for kind in ("disease", "chemical"):
                values.update(entity_values(ref[kind], generated[kind], context[kind], kind))
            if cfg["evaluation"].get("decision_mapper") == "terminal_keyword":
                values["decision_preserved_heuristic"] = float(decision_from_terminal(row.predicted_answer) == str(row.final_decision).lower())
            rows.append(values)
        item = pd.DataFrame(rows)
        # Semantic scoring needs the source answers; keep output columns unchanged.
        item["long_answer"] = pred.long_answer.fillna("")
        item["predicted_answer"] = pred.predicted_answer.fillna("")
        item = optional_semantic(item, cfg).drop(columns=["long_answer", "predicted_answer"])
        item["base_conditional_ppl"] = conditional_ppl(pred, cfg["model_name"], None, cfg["evaluation"]["max_length"], cfg["evaluation"].get("perplexity_batch_size", 1))
        if cfg["evaluation"].get("compute_frozen_g0_ppl"):
            item["frozen_g0_conditional_ppl"] = conditional_ppl(pred, cfg["model_name"], run_dir / "g0" / "adapter", cfg["evaluation"]["max_length"], cfg["evaluation"].get("perplexity_batch_size", 1))
        item.to_csv(run_dir / f"g{generation}_per_example_metrics.csv", index=False)
        summary = item.drop(columns=["pubid"]).mean(numeric_only=True).to_dict()
        summary.update({"generation": f"G{generation}", "n": len(item), "prediction_file": path.name,
                        "prediction_distinct_1": distinct(pred.predicted_answer.tolist(), 1),
                        "prediction_distinct_2": distinct(pred.predicted_answer.tolist(), 2),
                        "prediction_unique_answer_ratio": pred.predicted_answer.nunique() / len(pred)})
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(run_dir / "evaluation_summary.csv", index=False)
    print(f"Wrote per-example metrics and {run_dir / 'evaluation_summary.csv'}")


if __name__ == "__main__":
    main()
