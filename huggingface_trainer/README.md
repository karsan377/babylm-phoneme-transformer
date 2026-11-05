# BabyLM Hugging Face Trainer pipeline

This directory reproduces Samson Wanfranek's BabyLM Strict text-model
experiment without requiring access to the original Kubernetes account. It
contains data preparation, tokenizer training, a randomly initialized
Llama-style causal model, Hugging Face Trainer training, checkpoint resume,
word-milestone exports, a ten-configuration sweep, model selection, and the
official BabyLM evaluation entry point.

The model is **not GPT-2 and not a pretrained Llama checkpoint**. It is a
35,662,336-parameter Llama-style decoder initialized from random weights and
trained with `transformers.Trainer`.

## Model and experiment

| Property | Value |
|---|---:|
| Hidden size | 512 |
| Transformer layers | 10 |
| Attention heads | 8 |
| Key/value heads | 4 |
| MLP size | 1,536 |
| Context | 512 tokens |
| Tokenizer | custom 8,192-token byte-level BPE |
| Effective batch size | 32 |
| Training epochs per sweep run | 10 |
| Maximum Strict exposure | 100 million words |
| Selection criterion | held-out validation loss |

The winning run used learning rate `1e-4`, cosine scheduling, weight decay
`0.01`, and warmup ratio `0.05`. Its validation perplexity was `31.3746`.

## 1. Installation

Python 3.11 and a CUDA-capable GPU are recommended. CPU execution works for
the smoke test but full training and evaluation are expensive.

```bash
cd huggingface_trainer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
python scripts/validate_setup.py --skip-data-check
```

`requirements.txt` contains the intended direct dependency ranges;
`requirements-lock.txt` records the versions used for reproducibility.

## 2. Obtain the data

Download the official BabyLM 2026 Strict text corpus following the challenge
instructions. Do not commit the corpus to Git. Place the training and
validation text files anywhere locally, then run:

```bash
python scripts/prepare_data.py \
  --train data/raw/train/*.train \
  --validation data/raw/dev/*.dev \
  --output-dir data/processed \
  --vocab-size 8192
```

The preparation script trains the tokenizer using training data only and
creates:

```text
data/processed/
├── manifest.json
├── tokenizer/
├── train_ids.npy
├── train_word_ends.npy
├── validation_ids.npy
└── validation_word_ends.npy
```

The manifest records source hashes, token counts, and word counts. Validate it
before expensive training:

```bash
python scripts/validate_setup.py --data-dir data/processed
```

If the official corpus uses different file extensions, pass its actual file
paths to `--train` and `--validation`; the script does not depend on the
extension.

## 3. Fast local smoke test

Use the included fixture text and prepare a small vocabulary:

```bash
python scripts/prepare_data.py \
  --train tests/fixtures/train.txt \
  --validation tests/fixtures/validation.txt \
  --output-dir data/smoke-processed \
  --vocab-size 256
python scripts/run_from_config.py configs/smoke.yaml
```

This initializes the real architecture, trains five optimizer steps, saves a
checkpoint, evaluates it, and exports a reloadable Hugging Face model. It is a
pipeline test, not a meaningful language model.

## 4. Train one full model

```bash
python scripts/run_from_config.py configs/train_model_b.yaml
```

Or call the canonical trainer directly:

```bash
python scripts/train.py \
  --data-dir data/processed \
  --output-dir outputs/model-b \
  --epochs 10 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --lr-scheduler cosine \
  --batch-size 8 \
  --gradient-accumulation 4 \
  --resume-from-checkpoint auto
```

`--resume-from-checkpoint auto` finds the newest valid Trainer checkpoint.
Training refuses to exceed the 100-million-word Strict budget. Model-only
exports are saved at 1–10M words and then every 10M words through 90M.

## 5. Run the ten-configuration sweep

```bash
python scripts/run_sweep.py \
  --data-dir data/processed \
  --output-dir outputs/sweep \
  --epochs 10
```

Each run is restart-safe. Completed trials are skipped and incomplete trials
resume from their newest checkpoint. The configurations vary learning rate,
scheduler, weight decay, and warmup. `sweep_summary.json` ranks completed runs
by validation loss.

For a cheap pipeline check, run the same ten configurations for five steps:

```bash
python scripts/run_sweep.py \
  --data-dir data/smoke-processed \
  --output-dir outputs/sweep-smoke \
  --epochs 1 --max-steps 5
```

Select and optionally export the winner:

```bash
python scripts/select_best_model.py --sweep-dir outputs/sweep
python scripts/select_best_model.py \
  --sweep-dir outputs/sweep \
  --export-dir models/model-b-best
```

## 6. Run official BabyLM evaluation

The repository includes the BabyLM evaluator in `../babylm-eval-2026`. First
download its evaluation data according to its README:

```bash
cd ../babylm-eval-2026/strict
pip install -r requirements.txt
python scripts/download_evals.py
cd ../../huggingface_trainer
```

Then run the compatibility preflight and full evaluation:

```bash
bash evaluation/run_full_eval.sh \
  models/model-b-best \
  ../babylm-eval-2026/strict
```

The runner performs zero-shot evaluation, reading-time evaluation,
GlobalPIQA, and all seven fine-tuning tasks. It writes `EVALUATION_COMPLETE`
only after every required fine-tuning result and prediction file exists.

The reported official scores use the upstream evaluator. Our separate
surprisal diagnostic identified a missing-leading-space issue for byte-level
tokenizers; diagnostic corrections must not be presented as official scores.

Create a compact CSV after evaluation:

```bash
python scripts/summarize_results.py \
  --result-root ../babylm-eval-2026/strict/results/model-b-best/main \
  --output results/generated/model-b-evaluation.csv
```

## 7. Optional Docker and Kubernetes execution

Docker uses the same commands as local execution:

```bash
docker build -t babylm-hf-trainer .
docker run --gpus all --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/outputs:/app/outputs" \
  babylm-hf-trainer \
  python scripts/run_from_config.py configs/smoke.yaml
```

Generic Kubernetes templates are under `kubernetes/`. They intentionally
contain placeholders instead of the original namespace, PVC, registry, or
credentials. TAs do not need our Kubernetes account: they can run locally or
substitute resources from their own cluster.

Never commit kubeconfig files, API tokens, service-account tokens, `.env`
files, or Kubernetes Secrets.

## 8. Tests

```bash
pytest
```

See [docs/reproduction.md](docs/reproduction.md) for the experiment tiers and
[docs/results.md](docs/results.md) for the recorded results.
