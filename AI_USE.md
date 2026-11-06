# AI Use Documentation

## Overview

We used ChatGPT (OpenAI) throughout the BabyLM project to help understand
unfamiliar tools, debug training and evaluation errors, and organize the final
repository. We also used it for guidance on Hugging Face Trainer, tokenizer
training, Kubernetes jobs, hyperparameter sweeps, and the official BabyLM
evaluation pipeline.

Most interactions involved asking technical questions, sharing code or terminal
output, and asking for help diagnosing unexpected behavior. We reviewed the
suggestions and tested them through local checks, Kubernetes runs, saved
training metrics, and evaluation reports.

The prompts below are condensed versions of the main technical questions. Minor
wording, spelling, repeated status questions, and debugging follow-ups were
cleaned up for readability.

## 1. Building the Hugging Face Model

We trained a Llama-style causal language model from scratch using Hugging Face
Trainer rather than using a pretrained model.

**Prompts:**

- How can we train a language model from scratch using Hugging Face Trainer?
- What architecture would be stronger than our earlier PicoTrainer model while
  remaining feasible under the BabyLM Strict data limit?
- How can we calculate validation perplexity from evaluation loss?
- How can training automatically resume from the latest checkpoint after a
  Kubernetes job failure?

ChatGPT helped explain how to initialize `LlamaForCausalLM` from a custom
configuration, connect it to Trainer, calculate perplexity, and restore Trainer,
optimizer, scheduler, and random-number-generator state.

We selected the architecture and training settings, ran the jobs, monitored the
results, and reviewed the generated implementation. The final model contains 10
transformer layers, a hidden size of 512, eight attention heads, and a
512-token context window.

## 2. Dataset Preparation and Tokenization

We trained a custom byte-level BPE tokenizer on the BabyLM training corpus and
prepared packed token arrays for efficient training.

**Prompts:**

- How can we train a byte-level BPE tokenizer only on the training data?
- How should we pack tokenized text into fixed-length sequences?
- How can we count cumulative training words instead of only tokens or steps?
- How can we guarantee that training stays below the 100-million-word Strict
  limit?

ChatGPT helped develop the preprocessing workflow, data manifest, source-file
hashes, token arrays, and word-ending arrays. The word-ending arrays allow each
optimizer step to be mapped to the number of corpus words seen by the model.

We inspected the tokenizer and dataset counts before running full training. The
training script checks the planned word exposure and stops before training if a
configuration would exceed the Strict limit.

## 3. Hyperparameter Sweep and Checkpoints

We ran ten configurations to compare learning rate, scheduler, weight decay,
and warmup ratio.

**Prompts:**

- Does one sweep run correspond to one hyperparameter combination?
- How should we design ten useful configurations while keeping the model, data,
  and seed fixed?
- How can completed runs be skipped while incomplete runs resume safely?
- How can we save BabyLM model checkpoints at specific word milestones?

ChatGPT helped organize the restart-safe sweep and implement a callback that
saves model-only checkpoints at 1M through 10M words and then every 10M words
through 90M.

We completed the ten training runs and selected the best model using held-out
validation loss rather than official BabyLM test results. The best run used a
learning rate of `1e-4`, cosine scheduling, weight decay of `0.01`, and a warmup
ratio of `0.05`.

## 4. Running Jobs on Kubernetes

Full training and evaluation were performed on a Kubernetes cluster because
they required more GPU time than was practical locally.

**Prompts:**

- How can we request an NVIDIA A10, L4, RTX 3090, or RTX 4090 instead of waiting
  for only one GPU type?
- How can we inspect job status, pod events, logs, and mounted storage?
- Why did a job fail after waiting for a GPU, and how can it be safely requeued?
- How can the repository remain reproducible without sharing our Kubernetes
  account or credentials?

ChatGPT helped interpret pod events and logs, distinguish scheduling delays from
application errors, and revise job configurations. We monitored each job and
verified its outputs on persistent storage.

The public Kubernetes files are templates only. They do not contain our
namespace, PVC names, kubeconfig, service-account tokens, or other credentials.
The same Python pipeline can be run locally by a teaching assistant.

## 5. Evaluating the Model

We evaluated the selected model using the official BabyLM strict evaluation
pipeline.

**Prompts:**

- How can we verify that our Hugging Face model and tokenizer are compatible
  with the BabyLM evaluator?
- How can we run zero-shot, reading-time, GlobalPIQA, and fine-tuning tasks?
- How can we prevent a silent evaluation error from being treated as successful
  completion?
- Why did evaluation surprisal differ from PicoTrainer surprisal?

ChatGPT helped develop a compatibility preflight and a runner that checks for
all required reports before creating an evaluation-complete marker. It also
helped investigate the reading-surprisal difference, which revealed a
leading-space issue for byte-level tokenizers.

We reviewed the official result files and kept the separate surprisal
diagnostic distinct from the official scores. We also used the same pipeline to
compare our model with the official BabyLM GPT-2 baseline.

## 6. Repository Organization and Reproducibility

We used ChatGPT to help convert the experimental files into a standalone branch
that teaching assistants can inspect and run.

**Prompts:**

- Which files are necessary to reproduce preprocessing, training, the sweep,
  model selection, and evaluation?
- How can we provide a fast local smoke test instead of requiring graders to
  repeat every full GPU experiment?
- How should hard-coded Kubernetes paths and credentials be removed?
- How can the Hugging Face branch be separated from the unrelated phoneme
  transformer project?

The final branch includes relative local paths, pinned dependencies, a
Dockerfile, generic Kubernetes templates, fixture data, tests, recorded result
summaries, and a README with reproduction commands. Large datasets, generated
models, output directories, and credentials are excluded from Git.

## 7. Testing AI-Assisted Code

We did not treat generated code as correct without testing it. The following
checks were used:

- Python files were compiled with `python -m py_compile`.
- Shell scripts were checked with `bash -n`.
- Training and Kubernetes YAML files were parsed with PyYAML.
- Unit tests checked word alignment and cumulative word accounting.
- A real Hugging Face Trainer preflight completed an optimization step.
- A five-step end-to-end smoke run trained a tokenizer, prepared data, trained
  the real architecture, evaluated it, saved a checkpoint, and exported a
  reloadable model.
- Full Kubernetes runs were checked through logs, events, Trainer state,
  milestone checkpoint files, and final metrics.
- Evaluation completion required the expected reports and prediction files.
- The final repository was searched for credentials and personal cluster
  identifiers before publication.

## 8. Summary of AI and Human Contributions

Overall, ChatGPT served as a coding and debugging assistant. It was most useful
for explaining unfamiliar libraries, suggesting implementations, interpreting
errors, and organizing reproducibility documentation.

We were responsible for choosing the research direction, making the main model
and experiment decisions, providing and reviewing the data, running the
training and evaluation jobs, selecting the final model, verifying results, and
reviewing AI-assisted code and documentation.
