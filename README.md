# Transformer-Based Multilingual Multiword Expression Identification

Model comparison and error analysis on PARSEME 2.0.

This repository uses `xlm-roberta-base` as its main baseline for PARSEME 2.0
Subtask 1: multilingual verbal multiword expression identification. The same
pipeline can also fine-tune compatible Hugging Face token-classification
encoders for controlled model comparisons. It reads the official CUPT files
from a separate data checkout and writes CUPT predictions that can be scored
with the official evaluator.

## Scope

- Task: PARSEME 2.0 Subtask 1 MWE identification.
- Baseline model: XLM-RoBERTa token classification.
- Languages: `EGY, EL, FA, FR, GRC, HE, JA, KA, LV, NL, PL, PT, RO, SL, SR, SV, UK`.
- Training: the 16 languages with both `train.cupt` and `dev.cupt`.
- GRC: blind-test prediction only, using multilingual zero-shot transfer.

## Dissertation Archive

The dissertation and presentation materials are archived separately and are not
included in this code repository. Generated model and experiment artifacts are
also excluded from version control.

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

### 2a. Run E1 with official checkpoint selection

E1 keeps the baseline architecture and hyperparameters, but trains for at most
three epochs, evaluates once per epoch, selects the checkpoint with the official
16-language MWE macro-F1, and stops after one non-improving evaluation:

```bash
./scripts/train_e1_official.sh
```

The default seed is `13`. Override it without editing the script:

```bash
SEED=42 ./scripts/train_e1_official.sh
```

E1 writes to `artifacts/experiments/e1-official-seed<SEED>/` so it does not
overwrite E0. Important outputs are:

```text
best_model/                    selected checkpoint exported as a standalone model
experiment_config.json         command, parameters, versions, Git revisions and result
training_metadata.json         selected checkpoint and official evaluation history
official-evaluations/eval-*/   per-epoch CUPT predictions and official reports
```

The training-time token metrics remain in the log for diagnostics, but they do
not control checkpoint selection in E1.

### 2b. Run E2 with square-root language sampling

E2 changes only the training-language sampler. All model, loss, optimizer,
checkpoint-selection and seed settings remain the same as E1:

```bash
./scripts/train_e2_sqrt_sampling.sh
```

For a language with `n_l` training chunks, E2 targets an expected sampling
probability proportional to `sqrt(n_l)`. Equivalently, each chunk receives
weight `n_l^-0.5`. This gives smaller languages more exposure without the
aggressive duplication caused by fully uniform language sampling. Sampling is
with replacement and draws the original number of chunks per epoch, so E1 and
E2 have the same number of optimizer steps.

The run is written to
`artifacts/experiments/e2-sqrt-sampling-seed<SEED>/`. Its
`experiment_config.json` records the original chunk counts, per-language
weights, and expected E2 distribution.

### 2c. Run E3 with tempered class-weighted loss

E3 retains E2's square-root language sampling and adds one change: token labels
use a tempered inverse-frequency cross-entropy loss.

```bash
./scripts/train_e3_class_weighted.sh
```

Weights are computed from the expected label distribution after E2 sampling:

```text
weight(label) = clip(
  (mean_observed_label_count / label_count)^0.25,
  0.75,
  2.5
)
```

The fourth-root exponent and clipping prevent extremely rare labels from
dominating training. Labels absent from the training data receive the maximum
weight in metadata but do not contribute to the loss because they never occur
as gold targets. All other model and training settings remain identical to E2.

E3 writes to
`artifacts/experiments/e3-class-weighted-seed<SEED>/`. The exact effective
label counts and weights are recorded under `loss` in both experiment JSON
files.

### 2d. Tune the current XLM-R baseline

The completed seed-13 runs currently give:

| Run | Sampling alpha | Class-weight alpha | Learning rate | Official dev macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| E1 | 1.0 | 0.0 | `2e-5` | 0.6674 |
| E2 | 0.5 | 0.0 | `2e-5` | **0.6839** |
| E3 | 0.5 | 0.25 | `2e-5` | 0.6716 |

E2 is therefore the reference configuration for further tuning. E3 increased
recall but reduced precision enough to lower macro-F1, so class weighting stays
disabled during the next sweep.

Use the controlled tuning script for the next experiments. It keeps the model,
batch sizes, sequence length, stride, weight decay, warmup, standard
cross-entropy loss and official checkpoint selection fixed. Change one of
`SAMPLING_ALPHA` or `LEARNING_RATE` at a time:

```bash
DRY_RUN=1 SAMPLING_ALPHA=0.75 ./scripts/train_xlmr_tuning.sh
```

`DRY_RUN=1` prints the exact command and output path without training. First
screen the two missing sampling values for one epoch:

```bash
SAMPLING_ALPHA=0.75 EPOCHS=1 ./scripts/train_xlmr_tuning.sh
SAMPLING_ALPHA=0.25 EPOCHS=1 ./scripts/train_xlmr_tuning.sh
```

Treat these one-epoch runs only as screening results. Rerun the strongest
sampling value for three epochs, then compare learning rates while holding that
sampling value fixed:

```bash
BEST_ALPHA=0.75
SAMPLING_ALPHA="$BEST_ALPHA" EPOCHS=3 ./scripts/train_xlmr_tuning.sh
SAMPLING_ALPHA="$BEST_ALPHA" LEARNING_RATE=1e-5 ./scripts/train_xlmr_tuning.sh
SAMPLING_ALPHA="$BEST_ALPHA" LEARNING_RATE=3e-5 ./scripts/train_xlmr_tuning.sh
```

The existing E2 result supplies the `alpha=0.5`, `learning_rate=2e-5`,
`seed=13` reference. Once the best full configuration is known, test whether
the gain is stable with two additional seeds:

```bash
BEST_ALPHA=0.75
BEST_LR=2e-5
SAMPLING_ALPHA="$BEST_ALPHA" LEARNING_RATE="$BEST_LR" SEED=42 ./scripts/train_xlmr_tuning.sh
SAMPLING_ALPHA="$BEST_ALPHA" LEARNING_RATE="$BEST_LR" SEED=2026 ./scripts/train_xlmr_tuning.sh
```

Select configurations using official development macro-F1 only; do not tune on
`test.blind`. Output names encode alpha, learning rate, epoch count and seed,
and every run records its exact configuration in `experiment_config.json`.
The script refuses to reuse an existing output directory, preventing accidental
checkpoint mixing.

### 2e. Compare XLM-R with mDeBERTa-v3

The primary encoder comparison is the tuned XLM-R baseline against
`microsoft/mdeberta-v3-base`. The comparison script fixes every training
setting to the current XLM-R reference: language-sampling alpha `0.5`, learning
rate `2e-5`, three epochs, standard cross-entropy, batch sizes `8/16`, maximum
length `256`, stride `64`, and official development macro-F1 checkpoint
selection. All encoders are loaded as float32 so checkpoints distributed in
float16 do not silently train with unscaled FP16 gradients on Apple MPS. Only
the pretrained encoder changes.

Install the updated tokenizer dependencies, then run a short French smoke test
before committing to the full multilingual run:

```bash
python -m pip install -r requirements.txt
SMOKE_TEST=1 ./scripts/train_encoder_comparison.sh mdeberta
./scripts/train_encoder_comparison.sh mdeberta
```

The full mDeBERTa result is written to:

```text
artifacts/experiments/compare-mdeberta-v3-base-fp32-a0p5-lr2em5-ep3-seed13/
```

The existing XLM-R runs already provide the comparison values. Across seeds
`13`, `42`, and `2026`, the tuned XLM-R configuration has official development
macro-F1 scores `0.6839`, `0.6752`, and `0.7068`, with mean `0.6886` and sample
standard deviation `0.0163`. First compare mDeBERTa using seed `13`; only if it
is competitive should it be repeated with the other seeds:

```bash
SEED=42 ./scripts/train_encoder_comparison.sh mdeberta
SEED=2026 ./scripts/train_encoder_comparison.sh mdeberta
```

Literal `FacebookAI/roberta-base` is an English-only model and is therefore not
a fair main comparison for these 17 non-English test languages. It can be
checked as an explicitly labelled negative control, but a full run requires
confirmation to avoid spending about one training run on a mismatched model:

```bash
SMOKE_TEST=1 ./scripts/train_encoder_comparison.sh roberta
ALLOW_ENGLISH_ROBERTA=1 ./scripts/train_encoder_comparison.sh roberta
```

Use `DRY_RUN=1` with any command to inspect the exact model, hyperparameters,
experiment name, and output directory without downloading or training.

### 2f. Tune mDeBERTa-v3 independently

The seed-13 FP32 comparison completed successfully. Its best checkpoint was at
epoch 2 with official development macro-F1 `0.6801`. This is competitive with
the seed-13 XLM-R result (`0.6839`), but the XLM-R hyperparameters must not be
assumed to be optimal for mDeBERTa.

Use the dedicated tuning script to vary one factor at a time. The experiment
name records the language-sampling alpha, class-weight alpha and clipping
bounds, learning rate, epoch count, and seed:

```bash
DRY_RUN=1 ./scripts/train_mdeberta_tuning.sh
```

The scripts use the active `python` by default. If the required dependencies
are installed in another environment, select it explicitly, for example:

```bash
export PYTHON_BIN=/opt/miniconda3/bin/python
```

The seed-13 controlled tuning runs are complete:

| Run | Learning rate | Sampling alpha | Class-weight alpha | Clip | Best epoch | Precision | Recall | Official dev macro-F1 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| M0 reference | `2e-5` | 0.5 | 0.0 | inactive | 2 | 0.6567 | 0.7052 | 0.6801 |
| M1 | `1e-5` | 0.5 | 0.0 | inactive | 3 | 0.6361 | 0.6952 | 0.6643 |
| M2 | `2e-5` | 0.25 | 0.0 | inactive | 3 | 0.6803 | 0.6971 | **0.6886** |
| M3 | `2e-5` | 0.25 | 0.25 | 0.75-1.5 | 3 | 0.6434 | 0.6973 | 0.6693 |

M1 reduced macro-F1 by `0.0158`, so `learning_rate=1e-5` is rejected. M2
improved macro-F1 over M0 by `0.0085`, driven by a `0.0236` precision increase
despite a `0.0081` recall decrease. The current mDeBERTa configuration therefore
keeps `learning_rate=2e-5` and uses `language_sampling_alpha=0.25`.

M3 added a mild weighted loss to M2. It reduced macro-F1 by `0.0193`: recall
changed by only `+0.0002`, while precision fell by `0.0369`. Class weighting is
therefore rejected for mDeBERTa, and a wider clipping range is not warranted.
The selected configuration remains M2: standard cross-entropy, learning rate
`2e-5`, sampling alpha `0.25`, and three epochs.

The three M2 confirmation runs are complete:

| Seed | Best epoch | Precision | Recall | Official dev macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 3 | 0.6803 | 0.6971 | 0.6886 |
| 42 | 3 | 0.6809 | 0.7013 | 0.6909 |
| 2026 | 3 | 0.6517 | 0.7027 | 0.6762 |
| Mean (sample SD) | - | 0.6710 (0.0167) | 0.7004 (0.0029) | 0.6852 (0.0079) |

These are controlled repetitions of the selected M2 configuration, not
additional tuning runs. Keep the three individual scores visible and do not
combine results from different hyperparameter configurations.

Run a smoke test before any additional weighted-loss configuration when its
parameters change substantially:

```bash
SMOKE_TEST=1 CLASS_WEIGHT_ALPHA=0.25 ./scripts/train_mdeberta_tuning.sh
```

### 2g. Add and independently tune the mBERT multilingual baseline

Use `google-bert/bert-base-multilingual-cased` as the third encoder. First run
the short compatibility check, then the full seed-13 comparison with the same
training protocol:

```bash
SMOKE_TEST=1 ./scripts/train_encoder_comparison.sh mbert
./scripts/train_encoder_comparison.sh mbert
```

The fixed-protocol mBERT reference is complete:

| Seed | Precision | Recall | Official dev macro-F1 |
| ---: | ---: | ---: | ---: |
| 13 | 0.6553 | 0.6357 | 0.6454 |
| 42 | 0.6689 | 0.6487 | 0.6586 |
| 2026 | 0.6441 | 0.6249 | 0.6344 |
| Mean (sample SD) | 0.6561 (0.0124) | 0.6364 (0.0119) | 0.6461 (0.0121) |

These seed repetitions measure variance; they are not hyperparameter tuning.
Tune mBERT independently on seed `13`, changing one factor at a time. The
controlled seed-13 results are:

| Run | Learning rate | Sampling alpha | Class-weight alpha | Class-weight clip | Best epoch | Precision | Recall | Official dev macro-F1 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| B0 reference | `2e-5` | 0.5 | 0.0 | inactive | 3 | 0.6553 | 0.6357 | 0.6454 |
| B1 | `2e-5` | 0.25 | 0.0 | inactive | 3 | 0.6611 | 0.6328 | **0.6466** |
| B2 | `1e-5` | 0.25 | 0.0 | inactive | 3 | 0.6263 | 0.6254 | 0.6258 |
| B3 | `2e-5` | 0.25 | 0.25 | 0.75-1.5 | 3 | 0.6240 | 0.6362 | 0.6300 |

B1 changed only language-sampling alpha from `0.5` to `0.25`. Macro-F1
increased by just `0.0012`: precision increased by `0.0058`, while recall
decreased by `0.0029`. Treat `0.25` as the provisional sampling value, not as a
conclusive improvement.

B2 changed only the learning rate from `2e-5` to `1e-5`. Macro-F1 decreased by
`0.0208`, with precision down by `0.0348` and recall down by `0.0074` relative
to B1. Reject `1e-5` and keep `2e-5`.

B3 returned to B1 and tested one mild weighted-loss configuration. Macro-F1
decreased by `0.0166`: recall increased by only `0.0034`, while precision fell
by `0.0371`. Reject class weighting. The selected seed-13 configuration is B1:
standard cross-entropy, learning rate `2e-5`, sampling alpha `0.25`, and three
epochs. Its `0.0012` advantage over B0 is small, so confirm B1 with the two
remaining seeds before drawing a final conclusion.

The B1 confirmation runs are complete:

| Seed | Best epoch | Precision | Recall | Official dev macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 3 | 0.6611 | 0.6328 | 0.6466 |
| 42 | 3 | 0.6623 | 0.6352 | 0.6485 |
| 2026 | 3 | 0.6633 | 0.6289 | 0.6456 |
| Mean (sample SD) | - | 0.6622 (0.0011) | 0.6323 (0.0032) | 0.6469 (0.0015) |

The tuned mean is only `0.0008` above the fixed-protocol mBERT mean (`0.6461`),
so the sampling change did not produce a material average improvement. B1 is
retained as the selected mBERT configuration because it was chosen using the
seed-13 tuning run before the confirmation seeds were evaluated.

### 2h. Select the final encoder

Using the same official development macro metric and the same three seeds, the
selected configuration for each encoder gives:

| Encoder | Sampling alpha | Class weighting | Learning rate | Precision mean (sample SD) | Recall mean (sample SD) | Official macro-F1 mean (sample SD) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| XLM-R base | 0.5 | Off | `2e-5` | 0.6957 (0.0257) | 0.6819 (0.0077) | **0.6886 (0.0163)** |
| mDeBERTa-v3 base | 0.25 | Off | `2e-5` | 0.6710 (0.0167) | 0.7004 (0.0029) | 0.6852 (0.0079) |
| mBERT base cased | 0.25 | Off | `2e-5` | 0.6622 (0.0011) | 0.6323 (0.0032) | 0.6469 (0.0015) |

XLM-R has the highest mean official macro-F1 and is selected as the primary
encoder. Its margin over mDeBERTa is only `0.0034`, so report mDeBERTa as a
close comparison rather than claiming a statistically significant difference
from three seeds. mBERT trails XLM-R by `0.0417` and serves as the lower-capacity
multilingual baseline. Per-language and error analysis is completed below
before creating the final blind-test prediction.

### 2i. Run per-language and error analysis

The analysis manifest fixes the nine selected checkpoints: three seeds for
each of XLM-R, mDeBERTa-v3, and mBERT. Generate the per-language summaries,
official error breakdowns, and qualitative false-positive/false-negative
examples with:

```bash
PYTHONPATH=src python -m mwe_roberta.cli analyze
```

Outputs are written to `artifacts/analysis/final-models/`. The main report is
`report.md`; the accompanying CSV files retain seed-level results, language
support counts, category/continuity/length/seen-status breakdowns, and
sentence-level qualitative examples. The analysis validates its custom CUPT
span extraction against the official TP/predicted/gold counts for every
language before writing results.

Across the 16 languages, XLM-R has the highest mean F1 on 10, mDeBERTa-v3 on
6, and mBERT on none. For XLM-R, the dominant generalization gap is between
seen and unseen expressions (pooled micro-F1 `0.8934` versus `0.3110`).
Discontinuous expressions also trail continuous ones (`0.6808` versus
`0.7990`), while single-token predictions have high recall but low precision
(`0.8710` recall, `0.3680` precision). Treat NL, PT, and EGY comparisons as
descriptive because their development sets contain only 8, 13, and 16 gold
MWEs, respectively.

### 2j. Evaluate the selected model on the blind benchmark

The [PARSEME 2.0 shared-task report](https://aclanthology.org/2026.mwe-1.33/)
publishes this blind-test ranking for the leading full-coverage systems:

| System | Test languages | Official test MWE-based macro-F1 |
| --- | ---: | ---: |
| MTLB-STRUCT | 17/17 | 0.5729 |
| Sahara-Tokenizers | 17/17 | 0.4839 |
| IPN | 17/17 | 0.2840 |
| baseline-gpt-oss-120b | 17/17 | 0.2325 |

These scores were not directly comparable with the XLM-R development mean
(`0.6886`), which uses the development sets of the 16 supervised languages.
The selected seed-13 XLM-R model has now also been evaluated on the hidden test
sets of all 17 languages, including zero-shot Ancient Greek, through the
official continuous [Codabench competition](https://www.codabench.org/competitions/13186/).

Submission `884918`, named `XLM-R-base-a0.5-lr2e-5-seed13`, obtained:

| Official blind-test measure | Precision | Recall | F1 | Leaderboard rank at submission |
| --- | ---: | ---: | ---: | ---: |
| Exact MWE-based | 0.6257 | 0.5969 | **0.6110** | **2/18** |
| Token-based | 0.7063 | 0.6217 | **0.6613** | **1/18** |

At the captured leaderboard snapshot, exact MWE F1 was `0.0108` below the
leader and `0.0381` above MTLB-STRUCT. The leaderboard is continuous, so these
ranks can change as new submissions arrive.

The matched-language comparison separates ordinary development-to-test shift
from the zero-shot language effect:

| Evaluation scope | Languages | Exact MWE macro-F1 |
| --- | ---: | ---: |
| Development, seed 13 | 16 supervised | 0.6839 |
| Blind test, same languages | 16 supervised | 0.6458 |
| Blind test, complete benchmark | 17, including GRC | 0.6110 |

The first drop is `0.0381`; adding GRC, whose zero-shot test F1 is `0.0436`,
reduces the complete macro score by a further `0.0348`. The largest supervised
language drops are KA, EL, EGY, SV, and FR. Continuous and multi-token MWEs
show the largest phenomenon-level declines, while unseen-in-training test F1
remains low at `0.3170`.

Reproduce the downloaded-result analysis with:

```bash
PYTHONPATH=src python -m mwe_roberta.cli analyze-benchmark
```

The full report is written to
`artifacts/analysis/codabench-884918/report.md`; its CSV exports preserve the
global, per-language, phenomenon, and per-category official results.

Package an existing prediction directory with the required top-level language
folders as follows:

```bash
./scripts/package_codabench_submission.sh \
  artifacts/predictions/e2-sqrt-sampling-seed13
```

The script verifies that every language contains `test.system.cupt`, refuses
to overwrite an existing archive, and writes the submission under
`artifacts/submissions/`.

### 2k. Run the preliminary local LLM comparison

The preliminary prompted-LLM experiment uses the locally cached post-trained
`Qwen/Qwen3.5-4B` checkpoint. It predates the independent LLM search below:
four positive and four negative development sentences are selected for each of
the 16 supervised languages, and the model receives four fixed training-set
demonstrations per language. The final seed-13 XLM-R predictions are selected
at the same sentence indices so both systems are scored on exactly the same
128 sentences.

Prepare and inspect the frozen sample without loading the LLM:

```bash
./scripts/run_llm_baseline.sh --prepare-only
```

Run offline inference and official evaluation after the model is present in
the Hugging Face cache:

```bash
./scripts/run_llm_baseline.sh
```

The protocol uses non-thinking mode, greedy decoding, a 2,048-token input
limit, a 96-token output limit, and a strict JSON token-index schema. Invalid
groups are dropped with an audit warning; an unparseable response is retained
and scored as an empty prediction. The sample, prompts, raw responses, parsed
groups, model revision, package versions, CUPT files, and official reports are
stored under:

```text
artifacts/experiments/llm-qwen3p5-4b-4shot-seed13/
```

The completed matched-subset result is:

| System | MWE P | MWE R | MWE F1 | Token P | Token R | Token F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B, four-shot | 0.0292 | 0.0385 | 0.0332 | 0.1362 | 0.1541 | 0.1446 |
| Final XLM-R seed 13, same subset | 0.7653 | 0.6848 | **0.7228** | 0.8576 | 0.6905 | **0.7650** |

`126/128` Qwen responses were parseable, so its low exact score is mainly a
span-identification failure rather than an output-format failure. Only 3 of 89
predicted MWE groups exactly matched the 115 gold groups. This is an
exploratory comparison only: the balanced sample does not preserve natural
MWE prevalence, the XLM-R checkpoint was selected on the full development set,
and one 4B model with one prompt cannot represent LLMs generally. Public-data
contamination also cannot be excluded.

### 2l. Tune Qwen independently

The formal Qwen comparison selects its prompting and decoding configuration
using only a balanced training-set calibration sample. Calibration targets are
fixed before demonstrations are selected, and their sentence indices are
excluded from the demonstration pool. Five pre-registered candidates compare
0/2/4-shot prompting, greedy versus Qwen-recommended sampling, and thinking
versus non-thinking mode:

```bash
./scripts/tune_qwen_llm.sh
```

The command is resumable. It writes the fixed plan before inference, verifies
that every candidate uses the same 128 calibration sentences, selects by
official exact MWE macro-F1, freezes the winning demonstrations, and evaluates
the winner on a separately sampled seed-42 development subset. The completed
search selected four-shot, non-thinking, greedy decoding:

| Candidate | Train-calibration MWE F1 |
| --- | ---: |
| 4-shot, non-thinking, greedy | **0.0982** |
| 2-shot, non-thinking, greedy | 0.0954 |
| 4-shot, non-thinking, sampling | 0.0662 |
| 0-shot, non-thinking, greedy | 0.0260 |
| 4-shot, thinking, sampling | 0.0000 |

On the frozen seed-42 development subset, selected Qwen obtained exact MWE F1
`0.0334`, while final XLM-R obtained `0.7295` on the same 128 sentences. The
near-equality to the preliminary seed-13 Qwen result (`0.0332`) shows that the
train-only search confirmed the original configuration but did not yield a
generalising accuracy improvement. Full plans, raw generations, official
scores, and the final report are under:

```text
artifacts/experiments/llm-qwen3p5-4b-tuning-seed13/
```

### 2m. Fine-tune Qwen with LoRA

The prompted Qwen search above changes only the input prompt and decoding
configuration; it does not update model weights. The LoRA experiment is a
separate parameter-efficient fine-tuning condition. It keeps the same
token-index JSON task formulation, masks prompt tokens from the causal-language-
model loss, and updates low-rank adapters on the text stack while the base Qwen
weights remain frozen.

Run the one-step memory and compatibility check before any long experiment:

```bash
SMOKE_TEST=1 ./scripts/train_qwen_lora.sh
```

The smoke test is not a result. Before the 500-sentence-per-language final run,
screen the four pre-registered
resource-bounded candidates one at a time. Each candidate trains on the same
deterministic 100-sentence-per-language sample and is then scored on the same
seed-42 balanced development subset:

```bash
./scripts/tune_qwen_lora.sh L0  # rank 16, LR 2e-4, sampling alpha 0.5
./scripts/tune_qwen_lora.sh L1  # rank 16, LR 1e-4, sampling alpha 0.5
./scripts/tune_qwen_lora.sh L2  # rank 8,  LR 2e-4, sampling alpha 0.5
./scripts/tune_qwen_lora.sh L3  # rank 16, LR 2e-4, sampling alpha 0.25
```

The frozen plan is recorded in `configs/qwen_lora_protocol.json`. The completed
screening used identical 128-sentence manifests and produced:

| Candidate | Rank | LR | Sampling alpha | MWE F1 | Token F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L0 | 16 | 2e-4 | 0.5 | 0.2224 | **0.3694** |
| L1 | 16 | 1e-4 | 0.5 | 0.1807 | 0.3178 |
| L2 | 8 | 2e-4 | 0.5 | **0.2276** | 0.3362 |
| L3 | 16 | 2e-4 | 0.25 | 0.2109 | 0.3325 |

The pre-specified exact-MWE macro-F1 rule therefore selected L2. Its lead over
L0 is only 0.0052 on a small balanced subset, so this is a configuration-
selection result rather than a final performance claim. The initial final
seed-13 command uses the selected L2 settings and the 500-sentence cap:

```bash
./scripts/train_qwen_lora.sh
```

Use `DRY_RUN=1` to inspect the resolved command without loading the model.

The resource-bounded default protocol uses at most 500 deterministically
selected training sentences per language (7,442 selected and 7,348 retained
after length filtering at seed 13), rank `8`, LoRA alpha `16`, dropout `0.05`,
learning rate `2e-4`, one epoch, completion-only loss, maximum length `640`, and
language sampling alpha `0.5`. A longest-example stress test established that
`640` fits the 24 GB M4 Pro, whereas `768` exceeds MPS memory.
The cap and the resulting training-data difference from the encoder experiments
must be disclosed in the dissertation. Every run writes the resolved source and
selected data counts, target modules, trainable-parameter count, package
versions, and training metrics beside the adapter. These defaults are a
starting configuration and must be selected or rejected using development
results, not blind-test labels.

First evaluate a cheap matched subset, then run the complete 16-language
development evaluation for the dissertation table:

```bash
./scripts/evaluate_qwen_lora.sh
FULL_DEV=1 ./scripts/evaluate_qwen_lora.sh
```

After the seed-13 protocol is frozen, run the two confirmation seeds one at a
time. The wrapper skips completed phases, resumes interrupted training from the
latest retained checkpoint, and lets generation resume from `responses.jsonl`:

```bash
caffeinate -i ./scripts/run_qwen_lora_confirmation.sh 42
caffeinate -i ./scripts/run_qwen_lora_confirmation.sh 2026
```

For an unattended strictly sequential campaign, the equivalent command is:

```bash
caffeinate -i ./scripts/run_qwen_lora_confirmations.sh
```

Both training seeds use the same frozen seed-42 matched subset for diagnostic
comparison and are then evaluated on all 20,163 development sentences. The
capped training composition is also held fixed with training-subset seed 13;
only adapter initialisation, training order, and the weighted sampler use the
confirmation seed. This makes the three-seed dispersion an optimisation-seed
comparison rather than a mixture of optimisation and data-subset variation.
Do not run the two commands concurrently on one MPS device. Set
`RUN_FULL_DEV=0` only when intentionally stopping after the inexpensive
matched-subset check.

The confirmation campaign is complete. All three adapters used the same 7,348
encoded training sentences. Complete-development results are:

| Seed | MWE P | MWE R | MWE F1 | Token P | Token R | Token F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 0.5020 | 0.3794 | 0.4322 | 0.5990 | 0.4569 | 0.5184 |
| 42 | 0.4858 | 0.3817 | 0.4275 | 0.5672 | 0.4436 | 0.4978 |
| 2026 | 0.4872 | 0.3792 | 0.4265 | 0.5437 | 0.4253 | 0.4773 |
| Mean +/- sample SD | **0.4917 +/- 0.0090** | **0.3801 +/- 0.0014** | **0.4287 +/- 0.0030** | **0.5700 +/- 0.0278** | **0.4419 +/- 0.0159** | **0.4978 +/- 0.0206** |

Regenerate the validated machine-readable and Markdown summaries with:

```bash
python -m mwe_roberta.lora_summary
```

The generated files are under
`artifacts/analysis/qwen-lora-confirmations/`. The small MWE-F1 standard
deviation shows that the complete-development exact score is stable across the
three tested optimization seeds; it does not remove the unequal training-data
and objective differences from the encoder comparison.

Generation is resumable through `responses.jsonl`. The full-development run
uses zero in-context demonstrations, greedy non-thinking decoding, and the same
official PARSEME evaluator as the encoder experiments. Only this full-set score
may be placed beside XLM-R, mDeBERTa-v3, and mBERT in the primary table. Keep
the 128-sentence prompt-only result in a separate matched-subset table because
its balanced sample is not directly comparable with full-development scores.

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

The evaluator passes only `train.cupt` to the official `--train` argument, so
the seen/unseen sections are not contaminated by development annotations.

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
