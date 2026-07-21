# PARSEME 2.0 XLM-RoBERTa Fine-tuning Pipeline

This repository fine-tunes `xlm-roberta-base` for PARSEME 2.0 Subtask 1: multilingual verbal multiword expression identification. It reads the official CUPT files from a separate data checkout and writes CUPT predictions that can be scored with the official evaluator.

## Scope

- Task: PARSEME 2.0 Subtask 1 MWE identification.
- Model: XLM-RoBERTa token classification.
- Languages: `EGY, EL, FA, FR, GRC, HE, JA, KA, LV, NL, PL, PT, RO, SL, SR, SV, UK`.
- Training: the 16 languages with both `train.cupt` and `dev.cupt`.
- GRC: blind-test prediction only, using multilingual zero-shot transfer.

## Repository Layout

Keep the code and official data as sibling directories:

```text
PycharmProjects/
|-- mwe_roberta_pipeline/       # this repository
|   |-- scripts/
|   |-- src/
|   `-- artifacts/              # generated locally; ignored by Git
`-- sharedtask-data/            # official data repository; not copied here
    `-- 2.0/subtask1/
        |-- EGY/ ... UK/
        `-- tools/
```

The scripts use `../sharedtask-data/2.0/subtask1` by default. The experiments in this project use official shared-task data at commit `7ca305d8784a73827600776ff3861ab96ed4061b`.

For a different layout, set the data path explicitly:

```bash
export PARSEME_DATA_DIR=/absolute/path/to/sharedtask-data/2.0/subtask1
```

Generated files default to this repository's `artifacts/` directory. To place large checkpoints elsewhere:

```bash
export MWE_ARTIFACTS_DIR=/absolute/path/to/mwe-artifacts
```

## Label Scheme

The official `PARSEME:MWE` column uses sentence-local ids:

```text
1:VID   first token of MWE id 1, category VID
1       another token belonging to MWE id 1
*       not part of an MWE
```

The baseline converts these annotations to BIO labels:

```text
*       -> O
1:VID   -> B-VID
1       -> I-VID
```

For discontinuous MWEs, later occurrences of an id remain `I-*`. At prediction time, `B-*` starts an MWE id and `I-*` attaches to the latest active id of that category. This single-label baseline does not fully represent overlapping MWEs; when a token has multiple memberships, it uses the first official annotation.

## Install

Run installation again after moving the repository so the editable package points to its new location:

```bash
cd /path/to/mwe_roberta_pipeline
python -m pip install -r requirements.txt
python -m pip install -e .
```

The scripts also prepend the local `src/` directory to `PYTHONPATH`, so they always execute the code in the current checkout. The first training run needs access to the Hugging Face `xlm-roberta-base` weights unless they are already cached.

## Workflow

All commands below are run from the `mwe_roberta_pipeline` repository root.

### 1. Inspect data and build labels

```bash
./scripts/prepare_all.sh
```

Outputs:

```text
artifacts/metadata/labels.json
artifacts/metadata/data_summary.json
```

`GRC` should appear under `missing_train_or_dev`.

### 2. Train the multilingual baseline

Three-epoch run:

```bash
./scripts/train_all_languages.sh
```

Apple Silicon friendly one-epoch run:

```bash
./scripts/train_all_languages_fast.sh
```

The best models are saved under `artifacts/xlmr-all/best_model` and `artifacts/xlmr-all-1epoch/best_model`, respectively. Use the French smoke run to validate a new environment before starting a long job:

```bash
./scripts/train_fr_smoke.sh
```

### 3. Predict all 17 blind test sets

The default command uses the three-epoch model:

```bash
./scripts/predict_all_languages.sh
```

Pass another model directory as the first argument when needed:

```bash
./scripts/predict_all_languages.sh artifacts/xlmr-all-1epoch/best_model
```

Outputs are written to `artifacts/predictions/<LANG>/test.system.cupt`, including a zero-shot GRC prediction.

### 4. Predict and evaluate development data

```bash
python -m mwe_roberta.cli predict \
  --languages all \
  --split dev \
  --model-dir artifacts/xlmr-all-1epoch/best_model \
  --output-dir artifacts/predictions-dev

./scripts/evaluate_dev.sh
```

The official per-language reports and macro average are written to:

```text
artifacts/evaluation-dev/<LANG>/eval.txt
artifacts/evaluation-dev/average.txt
```

## Direct CLI Examples

The Python CLI uses the same default data and artifact locations as the scripts.

Train French only:

```bash
python -m mwe_roberta.cli train \
  --languages FR \
  --output-dir artifacts/xlmr-fr \
  --epochs 3
```

Predict French development data:

```bash
python -m mwe_roberta.cli predict \
  --languages FR \
  --split dev \
  --model-dir artifacts/xlmr-fr/best_model \
  --output-dir artifacts/predictions-fr-dev
```

Use `--data-dir`, `--output-dir`, and `--prediction-dir` to override individual commands without setting environment variables.

## GitHub Boundary

Commit the source code, scripts, dependency files, and documentation. Do not commit the official dataset, `artifacts/`, model checkpoints, predictions, `.idea/`, or generated `*.egg-info/` files. These paths are covered by `.gitignore`; existing local artifacts remain usable after the repository is initialized.

## Dissertation Experiments

- Use the multilingual model as the baseline.
- Report official per-language scores and macro averages.
- Treat GRC as zero-shot because this release has no GRC train/dev split.
- Useful comparisons include FR-only versus multilingual fine-tuning, base versus large models when GPU memory permits, and different sequence length or stride settings.
