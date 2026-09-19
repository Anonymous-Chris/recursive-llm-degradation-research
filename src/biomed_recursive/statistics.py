from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPARISONS = [(0, 1), (1, 2), (2, 3), (0, 3)]


def paired_bootstrap(delta: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    delta = delta[np.isfinite(delta)]
    means = np.empty(draws)
    for i in range(draws):
        means[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_effect(delta: np.ndarray) -> float:
    delta = delta[np.isfinite(delta)]
    sd = delta.std(ddof=1)
    return float(delta.mean() / sd) if len(delta) > 1 and sd else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap CIs and trajectory plots.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--metrics", required=True, help="comma-separated metric columns")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root, metrics, rng = Path(args.results_dir), args.metrics.split(","), np.random.default_rng(args.seed)
    output = root / "statistical_analysis"
    output.mkdir(exist_ok=True)
    comparisons, trajectories = [], []
    for condition_dir in root.iterdir():
        if not condition_dir.is_dir() or condition_dir.name == "statistical_analysis": continue
        for seed_dir in condition_dir.glob("seed_*"):
            summary = pd.read_csv(seed_dir / "evaluation_summary.csv")
            seed = seed_dir.name.removeprefix("seed_")
            for _, row in summary.iterrows():
                for metric in metrics:
                    if metric in row:
                        trajectories.append({"condition": condition_dir.name, "seed": seed, "generation": row.generation, "metric": metric, "value": row[metric]})
            tables = {g: pd.read_csv(seed_dir / f"g{g}_per_example_metrics.csv").set_index("pubid") for g in range(4)}
            for left, right in COMPARISONS:
                shared = tables[left].index.intersection(tables[right].index)
                for metric in metrics:
                    if metric not in tables[left] or metric not in tables[right]: continue
                    delta = (tables[right].loc[shared, metric] - tables[left].loc[shared, metric]).to_numpy(float)
                    finite = delta[np.isfinite(delta)]
                    if not len(finite): continue
                    low, high = paired_bootstrap(finite, args.bootstrap_draws, rng)
                    comparisons.append({"condition": condition_dir.name, "seed": seed, "metric": metric,
                                        "comparison": f"G{left}->G{right}", "n_paired": len(finite),
                                        "mean_difference": finite.mean(), "median_difference": np.median(finite),
                                        "sd_difference": finite.std(ddof=1), "ci95_low": low, "ci95_high": high,
                                        "paired_standardized_effect": paired_effect(finite)})
    pd.DataFrame(comparisons).to_csv(output / "paired_bootstrap_results.csv", index=False)
    trajectories = pd.DataFrame(trajectories)
    trajectories.to_csv(output / "seed_level_trajectories.csv", index=False)
    if not trajectories.empty:
        for (condition, metric), group in trajectories.groupby(["condition", "metric"]):
            group = group.copy(); group["g"] = group.generation.str.extract(r"(\d+)").astype(int)
            fig, ax = plt.subplots(figsize=(5, 3.3))
            for seed, seed_data in group.groupby("seed"):
                ax.plot(seed_data.g, seed_data.value, "o-", alpha=.35, label=f"seed {seed}")
            mean = group.groupby("g").value.agg(["mean", "std", "count"])
            sem = mean["std"].fillna(0) / np.sqrt(mean["count"])
            ax.errorbar(mean.index, mean["mean"], yerr=sem, color="black", linewidth=2, marker="o", label="mean ± SEM")
            ax.set(xticks=range(4), xticklabels=["G0", "G1", "G2", "G3"], xlabel="Generation", ylabel=metric,
                   title=f"{condition}: {metric}")
            ax.legend(fontsize=7); fig.tight_layout()
            fig.savefig(output / f"{condition}_{metric}.png", dpi=200); plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
