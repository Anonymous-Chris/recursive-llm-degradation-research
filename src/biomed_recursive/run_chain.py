from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import dump_json, load_config, normalize_table, set_seed, validate_splits
from .modeling import generate, train_adapter


def diagnostics(frame: pd.DataFrame, answer_col: str, max_tokens: int) -> dict:
    answers = frame[answer_col].fillna("").astype(str)
    lengths = answers.str.split().str.len()
    return {
        "rows": len(frame), "empty_answers": int((answers.str.strip() == "").sum()),
        "mean_words": float(lengths.mean()), "median_words": float(lengths.median()),
        "p95_words": float(lengths.quantile(.95)), "max_words": int(lengths.max()),
        "duplicate_answer_count": int(answers.duplicated().sum()),
        "unique_answer_ratio": float(answers.nunique() / max(1, len(answers))),
        "answers_at_or_above_max_new_tokens_approx": int((lengths >= max_tokens).sum()),
    }


def train_input(condition: str, generation: int, human: pd.DataFrame, prior: pd.DataFrame | None, anchor_ratio: float):
    if generation == 0 or condition == "human_control":
        return human.copy(), "human_train.csv"
    synthetic = prior.rename(columns={"synthetic_answer": "long_answer"}).copy()
    if condition == "recursive":
        return synthetic, f"g{generation - 1}_synthetic.csv"
    if condition == "anchor_10":
        count = round(len(human) * anchor_ratio)
        mixed = pd.concat([synthetic.iloc[:len(human) - count], human.iloc[:count]], ignore_index=True)
        return mixed.sample(frac=1, random_state=generation).reset_index(drop=True), f"g{generation - 1}_synthetic + human_anchor"
    raise ValueError("condition must be recursive, human_control, or anchor_10")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and generate one complete G0--G3 chain.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=["recursive", "human_control", "anchor_10"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    data_root = Path(cfg["data_root"])
    human, gold = normalize_table(pd.read_csv(data_root / "human_train.csv", dtype={"pubid": str})), normalize_table(pd.read_csv(data_root / "gold_eval.csv", dtype={"pubid": str}))
    if len(human) != cfg["g0_base_size"] or len(gold) != cfg["gold_eval_size"]:
        raise ValueError("Prepared CSV row counts do not match the declared experimental contract.")
    validate_splits(human, gold)
    run_dir = Path(cfg["output_root"]) / args.condition / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prior_synthetic, previous_adapter = None, None
    metadata = {"condition": args.condition, "seed": args.seed, "config": cfg, "stages": []}
    for generation in range(cfg["generations"] + 1):
        stage = run_dir / f"g{generation}"
        stage.mkdir(exist_ok=True)
        training, provenance = train_input(args.condition, generation, human, prior_synthetic, cfg["anchor_ratio"])
        training.to_csv(stage / "training_input.csv", index=False)
        init_adapter = previous_adapter if (generation and cfg["init_mode"] == "previous_adapter") else None
        adapter = train_adapter(training, cfg, stage, init_adapter, seed=args.seed)
        predictions = generate(gold, cfg, adapter, "predicted_answer")
        predictions.to_csv(run_dir / f"g{generation}_predictions.csv", index=False)
        item = {"generation": generation, "training_provenance": provenance, "init_adapter": str(init_adapter) if init_adapter else "base", "prediction_rows": len(predictions)}
        if generation < cfg["generations"]:
            synthetic = generate(human, cfg, adapter, "synthetic_answer")
            keep = ["pubid", "question", "context", "synthetic_answer", "final_decision"]
            synthetic[keep].to_csv(run_dir / f"g{generation}_synthetic.csv", index=False)
            item["synthetic_diagnostics"] = diagnostics(synthetic, "synthetic_answer", cfg["generation"]["max_new_tokens"])
            prior_synthetic = synthetic[keep]
        metadata["stages"].append(item)
        dump_json(run_dir / "run_metadata.json", metadata)
        previous_adapter = adapter
    print(f"Completed {run_dir}")


if __name__ == "__main__":
    main()
