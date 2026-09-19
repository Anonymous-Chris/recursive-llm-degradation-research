from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from .common import context_to_text, dump_json, load_config, set_seed, normalize_table, validate_splits


def to_frame(dataset) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(dataset):
        rows.append({
            "pubid": item.get("pubid", item.get("id", index)),
            "question": item.get("question", ""),
            "context": context_to_text(item.get("context", item.get("contexts", ""))),
            "long_answer": item.get("long_answer", ""),
            "final_decision": item.get("final_decision", ""),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create immutable PubMedQA study splits.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    root = Path(cfg["data_root"])
    root.mkdir(parents=True, exist_ok=True)

    # PQA-L has 1,000 expert-labelled rows. It is written once and never used
    # as input to run_chain. If a human training CSV was already supplied, do
    # not overwrite it with a guessed substitute.
    labeled = to_frame(load_dataset("pubmed_qa", "pqa_labeled", split="train"))
    if len(labeled) < cfg["gold_eval_size"]:
        raise ValueError("The labelled PubMedQA split is smaller than gold_eval_size.")
    gold = labeled.sample(n=cfg["gold_eval_size"], random_state=cfg["seeds"][0]).reset_index(drop=True)
    supplied = root / "human_train.csv"
    if supplied.exists():
        human = pd.read_csv(supplied, dtype={"pubid": str})
    else:
        unlabeled = to_frame(load_dataset("pubmed_qa", "pqa_unlabeled", split="train"))
        unlabeled = unlabeled[~unlabeled.pubid.astype(str).isin(gold.pubid.astype(str))]
        human = unlabeled.sample(n=cfg["g0_base_size"], random_state=cfg["seeds"][0]).reset_index(drop=True)
    human, gold = normalize_table(human), normalize_table(gold)
    validate_splits(human, gold)
    if len(human) != cfg["g0_base_size"] or (human.long_answer.fillna("").astype(str).str.strip() == "").any():
        raise ValueError(
            f"G0 requires exactly {cfg['g0_base_size']} non-empty long answers. Provide data/human_train.csv with "
            "pubid, question, context, long_answer, final_decision if using a custom human split."
        )
    gold.to_csv(root / "gold_eval.csv", index=False)
    human.to_csv(root / "human_train.csv", index=False)
    dump_json(root / "split_metadata.json", {
        "gold_rows": len(gold), "human_rows": len(human), "gold_pubids": gold.pubid.tolist(),
        "sampling_seed": cfg["seeds"][0], "source": "pubmed_qa",
    })
    print(f"Wrote {root / 'gold_eval.csv'} and {root / 'human_train.csv'}")


if __name__ == "__main__":
    main()
