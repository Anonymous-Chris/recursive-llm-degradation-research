from __future__ import annotations

import argparse
import subprocess
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch all preregistered chains sequentially.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for seed in cfg["seeds"]:
        for condition in ("recursive", "human_control", "anchor_10"):
            subprocess.run([sys.executable, "-m", "biomed_recursive.run_chain", "--config", args.config,
                            "--condition", condition, "--seed", str(seed)], check=True)
            subprocess.run([sys.executable, "-m", "biomed_recursive.evaluate", "--config", args.config,
                            "--run-dir", f"{cfg['output_root']}/{condition}/seed_{seed}"], check=True)


if __name__ == "__main__":
    main()
