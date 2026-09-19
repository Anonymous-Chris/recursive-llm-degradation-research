# Recursive biomedical QA study

This is a reproducible implementation of the proposed PubMedQA study.  It does
not include model outputs: those must be generated on the intended GPU so that
all reported results come from one documented run.

## Experimental contract

* `PQA-U`: 500 examples for the original human training set.
* `PQA-L`: a fixed 1,000-example, expert-labelled evaluation set; it is never
  used as training data.
* Main condition: human -> G0 -> synthetic -> G1 -> synthetic -> G2 ->
  synthetic -> G3.
* Controls: repeated human-data fine-tuning and a 10% retained-human anchor.
* Every model in a chain is evaluated on exactly the same saved gold CSV.

The default `init_mode: previous_adapter` is **cumulative fine-tuning**:
G1 starts from G0, G2 from G1, and so on.  This makes the word "recursive"
refer to both the data and training trajectory.  Change it to `base` only if
the preregistered design is instead to re-train each generation from the same
base model; never mix the two designs in one table.

## Run order

From this directory, create an environment with the packages in
`requirements.txt`, log in to Hugging Face if necessary, then run:

```powershell
python -m biomed_recursive.prepare_data --config configs/experiment.yaml
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition recursive --seed 42
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition human_control --seed 42
python -m biomed_recursive.run_chain --config configs/experiment.yaml --condition anchor_10 --seed 42
python -m biomed_recursive.evaluate --config configs/experiment.yaml --run-dir results/recursive/seed_42
python -m biomed_recursive.statistics --results-dir results --metrics entity_disease_f1,entity_chemical_f1,rougeL,bertscore_f1,prediction_distinct_2,repetition_3gram,answer_tokens,base_conditional_ppl
```

Repeat the three chains and evaluation for seeds `123` and `456`.  The
convenience driver below launches the declared condition/seed matrix after the
gold split has been prepared:

```powershell
python scripts/run_matrix.py --config configs/experiment.yaml
```

`run_chain` writes adapters, synthetic data, prediction CSVs, generation
diagnostics and an immutable `run_metadata.json`. `evaluate` writes both
per-example measurements (needed for paired statistics) and a summary table.

## Outputs and checks

Before interpreting results, run:

```powershell
python scripts/verify_run.py --run-dir results/recursive/seed_42
```

It rejects a run unless every G0--G3 prediction file has exactly the saved
gold pubids in identical order, each synthetic set has the configured size,
and recursive training inputs have the expected provenance.  Do not compare
old pilot outputs generated with different decoding limits/settings.

### Metric interpretation

* Entity mismatch is reported as **reference-based entity mismatch**, never a
  hallucination rate. Context-supported and unsupported entity rates are
  reported separately.
* `distinct_1/2` are corpus-level diversity metrics for the synthetic data;
  3-gram repetition is an within-answer metric.
* Conditional perplexity is descriptive. Lower PPL is not evidence of better
  medical quality. Both a frozen base model and a frozen G0 evaluator are
  supported.
* Decision preservation is deliberately optional. Its mapper is a documented
  heuristic and must not be reported as clinician adjudication.

## Paper-safe scope

The supported conclusion is limited to observed changes under the documented
PubMedQA/model/decoding setup. This code cannot establish universal model
collapse or medical factuality without external clinical adjudication.
