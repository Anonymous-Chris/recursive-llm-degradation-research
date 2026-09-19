from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refuse malformed or test-mode recursive runs.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    if config["generations"] != 3: fail("expected NUM_GENERATIONS / generations = 3")
    expected_gold, expected_train = config["gold_eval_size"], config["g0_base_size"]
    data_root = Path(config["data_root"])
    def read(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    gold = read(data_root / "gold_eval.csv")
    human = read(data_root / "human_train.csv")
    for name, frame, expected in (("gold", gold, expected_gold), ("human", human, expected_train)):
        if len(frame) != expected or frame.pubid.eq("").any() or frame.pubid.duplicated().any():
            fail(f"Invalid {name} split size or pubids")
    if set(gold.pubid) & set(human.pubid):
        fail("Training and gold pubids overlap")
    reference = gold.pubid.tolist()
    if [stage["generation"] for stage in metadata["stages"]] != list(range(4)):
        fail("Missing or reordered stage metadata")
    prior = None

    def same(actual, expected, columns, label):
        if not actual[columns].reset_index(drop=True).equals(expected[columns].reset_index(drop=True)):
            fail(f"{label} differs from its source")

    for g in range(4):
        prediction = read(run_dir / f"g{g}_predictions.csv")
        if len(prediction) != expected_gold: fail(f"G{g} predictions have {len(prediction)}, expected {expected_gold}")
        ids = prediction.pubid.tolist()
        if ids != reference: fail(f"G{g} pubids differ from saved gold")
        same(prediction, gold, ["pubid", "question", "context", "long_answer"], f"G{g} gold inputs")
        training = read(run_dir / f"g{g}" / "training_input.csv")
        if len(training) != expected_train: fail(f"G{g} training input size changed")
        expected = human
        if g and metadata["condition"] != "human_control":
            expected = prior.rename(columns={"synthetic_answer": "long_answer"})
            if metadata["condition"] == "anchor_10":
                count = round(len(human) * config["anchor_ratio"])
                expected = pd.concat([expected.iloc[:len(human) - count], human.iloc[:count]], ignore_index=True)
                expected = expected.sample(frac=1, random_state=g).reset_index(drop=True)
        same(training, expected, ["pubid", "question", "context", "long_answer"], f"G{g} training")
        if g < 3:
            synthetic = read(run_dir / f"g{g}_synthetic.csv")
            if len(synthetic) != expected_train: fail(f"G{g} synthetic rows changed")
            same(synthetic, human, ["pubid", "question", "context"], f"G{g} synthetic inputs")
            prior = synthetic
    for stage in metadata["stages"]:
        if stage["generation"] and metadata["condition"] == "recursive" and "synthetic" not in stage["training_provenance"]:
            fail(f"G{stage['generation']} is not trained on prior synthetic data")
    print("PASS: fixed gold set, expected chain sizes, and documented provenance verified.")


if __name__ == "__main__":
    main()
