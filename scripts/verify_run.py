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
    reference = None
    for g in range(4):
        prediction = pd.read_csv(run_dir / f"g{g}_predictions.csv", dtype={"pubid": str})
        if len(prediction) != expected_gold: fail(f"G{g} predictions have {len(prediction)}, expected {expected_gold}")
        ids = prediction.pubid.tolist()
        if reference is None: reference = ids
        elif ids != reference: fail(f"G{g} pubids differ from G0")
        training = pd.read_csv(run_dir / f"g{g}" / "training_input.csv")
        if len(training) != expected_train: fail(f"G{g} training input size changed")
        if g < 3:
            synthetic = pd.read_csv(run_dir / f"g{g}_synthetic.csv")
            if len(synthetic) != expected_train: fail(f"G{g} synthetic rows changed")
    for stage in metadata["stages"]:
        if stage["generation"] and metadata["condition"] == "recursive" and "synthetic" not in stage["training_provenance"]:
            fail(f"G{stage['generation']} is not trained on prior synthetic data")
    print("PASS: fixed gold set, expected chain sizes, and documented provenance verified.")


if __name__ == "__main__":
    main()
