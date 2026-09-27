# Recursive biomedical QA study

This repository contains a PubMedQA recursive synthetic-data study, its model
generation pipeline, and Colab notebooks for evaluating saved runs. The study
evaluates Qwen2.5-0.5B and Qwen2.5-3B across four generations (G0-G3) using
random seeds 42 and 123.

Evaluation notebooks and saved analysis artifacts are organized under
`notebooks/evaluation_qwen2.5-0.5b/` and
`notebooks/evaluation_qwen2.5-3b/`.

## Study design

- `PQA-U`: 5,000 examples for the initial human training pool
- `PQA-L`: 1,000 labeled examples reserved for held-out evaluation
- Four generations, G0 through G3, with cumulative adapter fine-tuning
- Seeds 42 and 123
- Recursive and Human-Control training conditions
- The configured `init_mode: previous_adapter` means each generation starts from
  the previous generation's adapter
- Generation settings and model parameters are recorded in
  `configs/experiment.yaml`

Do not combine results from a run with a different initialization mode or
decoding configuration in the same analysis.

## Dataset Sources

The experiments use the PubMedQA dataset:

`qiaojin/PubMedQA`

Released dataset artifacts:

- Qwen2.5-0.5B: https://huggingface.co/datasets/chrislimbe/pubmedqa-recursive-llm-degradation-qwen2.5-0.5b
- Qwen2.5-3B: https://huggingface.co/datasets/chrislimbe/pubmedqa-recursive-llm-degradation-qwen2.5-3b

This repository contains generated and derived research artifacts and does not
redistribute the original PubMedQA dataset in its entirety.

## Load the datasets

```python
from datasets import load_dataset

REPO = "chrislimbe/pubmedqa-recursive-llm-degradation-qwen2.5-3b"

configs = [
    "seed42_recursive_synthetic_3b",
    "seed42_recursive_predictions_3b",
    "seed123_recursive_synthetic_3b",
    "seed123_recursive_predictions_3b",
    "seed42_human_control_synthetic_3b",
    "seed42_human_control_predictions_3b",
    "seed123_human_control_synthetic_3b",
    "seed123_human_control_predictions_3b",
]

datasets_3b = {}

for config in configs:
    datasets_3b[config] = load_dataset(REPO, config)
```

```python
for name, dataset in datasets_3b.items():
    print(f"\n{name}")

    split_name = list(dataset.keys())[0]
    df = dataset[split_name].to_pandas()

    print(f"Rows: {len(df)}")
    display(df.head(10))
```

## Run the generation pipeline

Create and activate a Python environment, install the dependencies from
`requirements.txt`, and authenticate with Hugging Face if required by the model
or dataset. From the repository root:

```bash
python -m biomed_recursive.prepare_data --config configs/experiment.yaml
```

### Recursive — Seed 42

```bash
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition recursive --seed 42
```

### Recursive — Seed 123

```bash
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition recursive --seed 123
```

### Human-Control — Seed 42

```bash
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition human_control --seed 42
```

### Human-Control — Seed 123

```bash
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition human_control --seed 123
```

Generated runs are written under `results/`.

Check a run's expected files and provenance with:

```bash
python scripts/verify_run.py --run-dir results/recursive/seed_42
```

`run_chain` saves the model adapters, generated data, predictions, and run
metadata. This checkout does not include the standalone
`biomed_recursive.evaluate`, `biomed_recursive.statistics`, or
`biomed_recursive.metrics` modules. Use the evaluation notebooks described
below to analyze exported run CSVs.

## Evaluate and analyze results

The combined evaluation notebooks are designed for Google Colab. Upload the
prediction and generation CSV files from the relevant runs, then run the
matching notebook from top to bottom:

- `notebooks/evaluation_qwen2.5-0.5b/combined_evaluation_seed42_0.5b_colab.ipynb`
- `notebooks/evaluation_qwen2.5-0.5b/combined_evaluation_seed123_0.5b_colab.ipynb`
- `notebooks/evaluation_qwen2.5-3b/combined_evaluation_seed42_3b_colab.ipynb`
- `notebooks/evaluation_qwen2.5-3b/combined_evaluation_seed123_3b_colab.ipynb`

The `cross_condition_cross_seed_master.ipynb` notebooks combine results across
conditions and seeds. Their `final_results/` directories contain the currently
saved tables, plots, and analysis outputs.

## Evaluation metrics

The evaluation includes:

- Disease F1
- Chemical F1
- Disease entity retention
- Chemical entity retention
- Context-supported entity rate
- BLEU-1
- BLEU-4
- ROUGE-L
- METEOR
- chrF++
- BERTScore F1
- Embedding cosine similarity
- Answer length (words)
- 3-gram repetition rate
- Conditional answer-only perplexity

Embedding cosine similarity is computed using:

`sentence-transformers/all-mpnet-base-v2`

Entity matching uses normalized surface-form matching within the corresponding
entity type. Context-supported rate is based on normalized entity surface-form
occurrence in the supplied biomedical context.

Perplexity and text-similarity metrics are descriptive metrics and are not
treated as direct measures of biomedical correctness or clinical safety.

## Statistical analysis

For each condition and seed, generation-level changes are evaluated across
G0-G3.

The analysis includes:

- Mean change
- Bootstrap 95% confidence intervals
- Paired Cohen's `dz`
- Two-sided Wilcoxon signed-rank tests

The primary descriptive contrast between conditions is the difference-in-change:

```text
ΔRH = ΔR − ΔH
```

where:

```text
ΔR = R_G3 − R_G0
ΔH = H_G3 − H_G0
```

The difference-in-change is calculated separately for each seed and then
summarized across the two seeds.

Because only two seeds are used for each model size, cross-condition and
cross-model comparisons are primarily descriptive within the tested
configurations.

The Wilcoxon signed-rank tests are within-condition generation-change tests.
They should not be interpreted as direct statistical tests of the
Recursive-versus-Human-Control difference-in-change.

## Results and research artifacts

The released research artifacts include derived outputs such as:

- Generated synthetic training data
- Model predictions
- Per-sample evaluation results
- Generation-level summaries
- Statistical analysis outputs
- Plotting data
- Tables and figures
- Model adapters

The released artifacts correspond to the configurations and runs used in the
reported analysis.

## Reproducibility

When reproducing or extending an experiment, keep the following consistent with
the reported run:

- Model
- Training data
- Evaluation data
- Random seed
- Generation number
- Training condition
- Adapter initialization mode
- LoRA/QLoRA configuration
- Training configuration
- Decoding configuration
- Evaluation configuration

Do not combine outputs generated with different initialization modes, generation
settings, or evaluation configurations in the same analysis.

The generated run metadata should be retained together with the corresponding
prediction and synthetic-data artifacts.

## Interpretation limits

Entity overlap measures agreement with reference answers; it is not a clinical
hallucination rate.

Entity retention measures the proportion of reference entities preserved in
the generated answer using the project's normalized surface-form matching rule.

The context-supported entity rate is based on normalized entity surface-form
occurrence in the supplied biomedical context. Context-unmatched entities are
not treated as definitive evidence of clinical hallucination.

Perplexity and text-similarity metrics are descriptive metrics, not direct
measures of biomedical correctness or medical safety.

Claims should be limited to the models, data, seeds, generations, and
generation settings represented in the analyzed runs.

The study does not establish universal model collapse or a general scaling law.
Differences between the 0.5B and 3B configurations should be interpreted as
observations within the tested configurations.
