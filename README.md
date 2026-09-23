# Recursive biomedical QA study

This repository contains a PubMedQA recursive synthetic-data study, its model
generation pipeline, and Colab notebooks for evaluating saved runs. The current
configuration targets Qwen2.5-3B. Evaluation notebooks and saved analysis
artifacts are organized under `notebooks/evaluation_qwen2.5-0.5b/` and
`notebooks/evaluation_qwen2.5-3b/`.

## Study design

- `PQA-U`: 5,000 examples for the initial human training pool
- `PQA-L`: 1,000 labeled examples reserved for held-out evaluation
- Four generations, G0 through G3, with cumulative adapter fine-tuning
- Seeds 42 and 123, as configured in `configs/experiment.yaml`
- Generation settings and model parameters are recorded in that config

The configured `init_mode: previous_adapter` means each generation starts from
the previous generation's adapter. Do not combine results from a run with a
different initialization mode or decoding configuration in the same analysis.

## Dataset Sources

The experiments use the PubMedQA dataset:
`qiaojin/PubMedQA`

Updated dataset artifacts:

- Qwen2.5-0.5B: https://huggingface.co/datasets/chrislimbe/pubmedqa-recursive-llm-degradation-qwen2.5-0.5b
- Qwen2.5-3B: https://huggingface.co/datasets/chrislimbe/pubmedqa-recursive-llm-degradation-qwen2.5-3b

This repository contains generated/derived research artifacts and does not
redistribute the original PubMedQA dataset in its entirety.

## Run the generation pipeline

Create and activate a Python environment, install the dependencies from
`requirements.txt`, and authenticate with Hugging Face if required by the model
or dataset. From the repository root:

```bash
python -m biomed_recursive.prepare_data --config configs/experiment.yaml
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition recursive --seed 42
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition human_control --seed 42
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition anchor_10 --seed 42
```

The configured matrix can be launched with:

```bash
python scripts/run_matrix.py --config configs/experiment.yaml
```

Generated runs are written under `results/`. Check a run's expected files and
provenance with:

```bash
python scripts/verify_run.py --run-dir results/recursive/seed_42
```

`run_chain` saves the model adapters, generated data, predictions, and run
metadata. This checkout does not include the standalone `biomed_recursive.evaluate`,
`biomed_recursive.statistics`, or `biomed_recursive.metrics` modules. Use the
evaluation notebooks described below to analyze exported run CSVs.

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
saved tables, plots, and (for the 0.5B analysis) a data archive. See the 3B
folder's README for its Colab upload and export workflow.

## Interpretation limits

Entity overlap measures agreement with reference answers; it is not a clinical
hallucination rate. Perplexity and text similarity are descriptive metrics, not
measures of medical safety or correctness. Claims should be limited to the
models, data, and generation settings represented in the analyzed runs.
