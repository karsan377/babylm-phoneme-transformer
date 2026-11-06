# Phoneme Transformers

Forked from [codebyzeb/PhonemeTransformers](https://github.com/codebyzeb/PhonemeTransformers), which contains the original training code and configs for transformers trained on phonemes. This fork adds Nautilus job files, pinned dependencies, and evaluation setups for BLiMP and EWoK.

Our mentor, Alex Warstadt, provisioned the Nautilus cluster for the group. All training and evaluation here ran on it.

## Setup

Reproduction used commit `302a9d3`.

```bash
git clone --recurse-submodules https://github.com/karsan377/babylm-phoneme-transformer.git
cd babylm-phoneme-transformer
git checkout 302a9d3
git submodule update --init --recursive
pip install -r requirements.txt
pip install ./submodules/evaluation-pipeline-2024
pip install "wandb==0.18.7"
```

Create a `.env` file in the repo root and source it before running anything:

```bash
export WANDB_ENTITY="your-wandb-entity"
export HF_READ_TOKEN="your-hf-token"
export HF_WRITE_TOKEN="your-hf-token"
```

Data and the tokenizer download from Hugging Face on first use:
- Tokenizer: `phonemetransformers/GPT2-85M-CHAR-PHON-SPACELESS`
- Training data: `phonemetransformers/IPA-BabyLM` (subset `strict_small`)
- Evaluation data: `phonemetransformers/IPA-BabyLM-evaluation`

## Weights & Biases

Training logs to W&B when `experiment.offline_run=False`. Each W&B project is the experiment group, so this project is `phoneme-strict-small`. Run `wandb login` once with your API key, and set `WANDB_ENTITY` to your team or username. Use `experiment.offline_run=True` to turn logging off.

Evaluation jobs don't log to W&B. Their scores are written to the checkpoint folder and the job logs.

## Training

Training needs a GPU. The reported run used one Tesla V100 (16 GB) and took about 13 hours.

```bash
python train.py \
  experiment=babylm_comparison_phoneme \
  dataset=babylm_strict_small_phoneme dataset.subconfig=strict_small \
  tokenizer=babylm_phoneme model=gpt2_85M \
  experiment.name=phoneme-strict-small-gpt2-85M experiment.group=phoneme-strict-small \
  experiment.offline_run=False experiment.push_to_hub=False \
  experiment.evaluate_babyslm=False experiment.blimp_tasks=null
```

Checkpoints are saved to `checkpoints/<group>/<name>/` every 50k steps. On Nautilus, `scripts/nautilus/phoneme-checkpoints-pvc.yaml` creates a volume for them, and `scripts/nautilus/phoneme-train-job.yaml` runs the same command in a pod.

## Evaluation

There are two evaluation setups.

**2024 data (what produced the results below).** BLiMP and EWoK are scored with the `evaluation-pipeline-2024` submodule on the phonemized 2024 data:
- `scripts/nautilus/phoneme-blimp-eval-job.yaml`
- `scripts/nautilus/phoneme-ewok-eval-job.yaml`

Both write `blimp_results.json` into the checkpoint folder, so run them one at a time. The second overwrites the first. Per-task tables are in the job logs.

**2026 data:** For this setup, the tokenizer wrapper phonemizes English text with espeak and maps it to the IPA-BabyLM phone inventory. The script runs BLiMP, the supplement, COMPS and reading. 

The code to run this is in the babylm-eval-2026 folder, sourced from the [2026 babylm-eval repo](https://github.com/babylm-org/babylm-eval/tree/main/strict). To run this, install the following:

```
pip install -r requirements.txt
pip install --upgrade --force-reinstall torch
apt-get install espeak-ng
```

In the strict folder, run:

`python -m scripts.download_evals`

And to run the evaluation, run:

`bash scripts/eval_full_phoneme.sh ../checkpoints/checkpoint-100000 causal 2>&1 | tee eval_logs.log`

Use a path to the specific checkpoint to run the evaluation for. The results will be printed both to terminal output and to a specified log file.

## Results

Chance is 50%. Full summary is in `results/RESULTS.md`.

2024 phonemized data: 

| Checkpoint | BLiMP (filtered) | BLiMP (supplement) | EWoK |
|---|---|---|---|
| 150k | 65.2% | 57.2% | 49.4% |
| 200k | 64.8% | 57.1% | 49.6% |

2026 phonemized data:

| Checkpoint | BLiMP (filtered) | BLiMP (supplement) | EWoK | COMPS | Reading
|---|---|---|---|---|---|
| 100k | 64.26% | 55.44% | 51.64% | 51.40% | 0.985 |

## Notes

- The sweep, dropout and size-comparison configs from the original repo are still here. We haven't rerun them on this fork.
- The original `scripts/sweep.yaml` and SLURM workflows aren't set up here.

## AI Acknowledgement

AI, specifically Claude Haiku 5.5, was used throughout this project for writing yaml configs, Nautilus job manifests, debugging (from software incompatability to cluster failures issues), and project direction. 
