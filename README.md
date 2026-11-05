# BabyLM 2026 — Hugging Face GPT-2

This branch contains a GPT-2 language model trained from scratch on the BabyLM 2026 Strict-Small dataset using Hugging Face Trainer.

The repository includes the pretrained model weights, tokenizer, and evaluation results. The pretrained model can be evaluated locally without access to Nautilus or Kubernetes.

## Model Architecture

| Parameter | Value |
|---|---|
| Architecture | GPT-2 (causal language model) |
| Transformer layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 256 |
| Context length | 256 |
| Vocabulary size | 50,257 |
| Training dataset | BabyLM 2026 Strict-Small |
| Training epochs | 10 |

## Training and Hyperparameter Sweep

The model was trained using Hugging Face Trainer, with Weights & Biases used to track a hyperparameter sweep.

The sweep explored:

- Learning rate: log-uniform between 1e-6 and 1e-4
- Gradient accumulation steps: 1–16
- Learning rate scheduler: linear or cosine

The model with the lowest recorded validation perplexity was selected for evaluation.

**Best run:** `neefqhar`

**Best validation perplexity:** 35.2958

Training results for all 10 runs are available in [`results/training_results.csv`](results/training_results.csv).

### Reproducing Training

The repository includes the original training and preprocessing scripts, along with the hyperparameter sweep configuration used for the experiment.

**1. Prepare the dataset**

Run the preprocessing script from the repository's root directory:

```bash
python prepare_data.py
```

This script downloads BabyLM 2026 Strict-Small, uses the GPT-2 tokenizer in `model/`, creates a reproducible 95% training / 5% validation split (seed 42), and packs the tokenized data into sequences of 256 tokens.

The resulting files are saved to:

- `data/train.jsonl`
- `data/val.jsonl`

**2. Hyperparameter sweep**

The sweep configuration is stored in `configs/sweep.yaml`. It specifies random search over learning rate, gradient accumulation steps, and learning rate scheduler.

To create a new sweep:

```bash
wandb sweep --project babylm-hf-hyperparameter-sweep configs/sweep.yaml
```

W&B will return a sweep ID. To launch a local sweep agent:

```bash
wandb agent ENTITY/PROJECT/SWEEP_ID
```

Replace the placeholders with your W&B entity, project, and newly created sweep ID.

**3. Nautilus training**

The `configs/sweep-job.yaml` file documents the Kubernetes job used to run the original sweep on Nautilus.

The original job launched 10 runs with up to 3 running in parallel. It references the original W&B sweep, Kubernetes Secret, and shared storage configuration, so these settings must be adapted before reuse.

**4. Model output**

The training script saves completed models to `trained_models/<run_id>/` by default.

To specify another output directory, set the `MODEL_OUTPUT_DIR` environment variable. The original Nautilus experiment saved models under `/shared/esther/models/`.

The pretrained model provided in `model/` corresponds to the selected best run, `neefqhar`.

## Loading the Pretrained Model

The `model/` directory contains the trained GPT-2 weights, configuration, and tokenizer files.

Install the required Python packages:

```bash
python -m pip install torch transformers
```

Load the model from the repository's root directory:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

model.eval()
```

No additional training is required to load the pretrained model.

## BabyLM Evaluation

This model was evaluated using the official [BabyLM evaluation pipeline](https://github.com/babylm-org/babylm-eval).

The repository includes results from both fast and full evaluations.

### Running the Fast Evaluation Locally

The pretrained model can be evaluated on a local machine without access to Nautilus or Kubernetes.

**1. Clone the official BabyLM evaluation repository:**

```bash
git clone https://github.com/babylm-org/babylm-eval.git
cd babylm-eval/strict
```

**2. Install the evaluation dependencies:**

```bash
python -m pip install -r requirements.txt
```

**3. Download the evaluation datasets:**

First, authenticate with Hugging Face:

```bash
huggingface-cli login
```

Then download the datasets:

```bash
python -m scripts.download_evals
```

The fast EWoK evaluation dataset is password-protected. To access the dataset, refer to the official BabyLM evaluation repository and follow the instructions provided there for downloading and extracting the EWoK dataset.

Refer to the official BabyLM README for any additional dataset setup requirements.

**4. Run the fast zero-shot evaluation:**

```bash
bash scripts/eval_zero_shot_fast.sh /absolute/path/to/babylm-phoneme-transformer/model gpt2-esther causal
```

Replace `/absolute/path/to/babylm-phoneme-transformer/model` with the actual path to the `model/` directory in this repository.

The `causal` backend is used because GPT-2 is a causal language model.

The fast evaluation uses a smaller subset of the evaluation benchmarks than the full evaluation, making it suitable for quicker testing.

## Evaluation Results

The model was evaluated on BabyLM benchmarks including BLiMP, BLiMP Supplement, EWoK, Entity Tracking, and reading tasks.

| Evaluation | Results |
|---|---|
| Fast zero-shot | [fast_eval_results.csv](results/fast_eval_results.csv) |
| Full zero-shot | [full_eval_results.csv](results/full_eval_results.csv) |
| Full fine-tuning | [finetuning_results.csv](results/finetuning_results.csv) |
| Hyperparameter sweep | [training_results.csv](results/training_results.csv) |
| Evaluation notebook | [results.ipynb](results/results.ipynb) |

The fast evaluation uses smaller evaluation datasets and is intended to provide a quicker way to assess model performance. The full evaluation provides more comprehensive benchmark results.

## Repository Structure

```text
model/
    config.json
    model.safetensors
    tokenizer files
    training_args.bin

results/
    fast_eval_results.csv
    full_eval_results.csv
    finetuning_results.csv
    training_results.csv
    results.ipynb

README.md
```

## Notes

- The pretrained weights correspond to the selected best hyperparameter sweep run.
- The model can be loaded and evaluated without retraining.
- The fast evaluation requires the official BabyLM evaluation pipeline and its associated evaluation datasets.

## AI and Human Contributions

ChatGPT was used to assist with understanding unfamiliar tools, debugging code, configuring training experiments, and organizing repository documentation. I was responsible for implementing and running the GPT-2 training pipeline, conducting hyperparameter sweeps, evaluating the trained model, and reviewing AI-assisted suggestions.

For a detailed summary of important prompts, AI contributions, and testing procedures, see AI_USE.md.