# BabyLM Strict-Small: hyperparameter sweep using pico-train

Pretrains small decoder-only models on BabyLM Strict-Small (10M words) and sweeps
learning rate (log-uniform 1e-6 to 1e-4), gradient accumulation (log-uniform 1..16) and LR schedule.

## Setup

pip install -r requirements.txt
Copy `.env.example` to `.env`, fill in your tokens, and load it (bash: `set -a; source .env; set +a`).
In `k8/job.yaml` and `k8/job2.yaml`, replace hardcoded values to your own credentials (see lines labeled `# currently hardcoded to my info`)

- HF_TOKEN: Hugging Face token (need to generate from your account)
- WANDB_API_KEY: Weights and Biases API key (need to generate from your account)
- HF_USER: Hugging Face username
- WANDB_ENTITY: Weights and Biases project name

## Run

1. python prep_data.py # tokenizes the data, pushes babylm-strict-small-512 to your HF account
2. python make_x10.py # 10 shuffled copies = 10 epochs
3. python sweep.py # needs a GPU; MB = micro-batch that fits your GPU (we used 8; 48 fit on 32 GB)

Results: runs/sweep-XX/val_ppl.csv, merged into sweep_val_ppl.csv. Plot with plot.py.

## Notes

- Tested on NVIDIA GPUs on the Nautilus cluster (k8s/ holds the job files; optional).
- Evaluation: github.com/babylm-org/babylm-eval (strict track).
- Trained models: <links to your Hugging Face repos>

## AI Contribution

AI, specifically Claude (Sonnet 5.5 Medium Free Plan) was utilized to write yaml configs and model modifications, Nautilus manifest, and debugging (fix dependency mismatch, resource errors). Additionally, was used to understand how to setup and what was running.

Example prompts from pico-train setup:

```
26-10-23 22:09:14 - pico-train - INFO - ├── Loss: 10.9424
2026-10-23 22:09:14 - pico-train - INFO - ├── Learning Rate: 9.82e-07
2026-10-23 22:09:14 - pico-train - INFO - └── Inf/NaN count: 0
2026-10-23 22:09:46 - pico-train - INFO - Step 100 -- 🔄 Training Metrics
2026-10-23 22:09:46 - pico-train - INFO - ├── Loss: 10.9335
2026-10-23 22:09:46 - pico-train - INFO - ├── Learning Rate: 1.96e-06
2026-10-23 22:09:46 - pico-train - INFO - └── Inf/NaN count: 0
2026-10-23 22:10:19 - pico-train - INFO - Step 150 -- 🔄 Training Metrics
2026-10-23 22:10:19 - pico-train - INFO - ├── Loss: 10.9128
2026-10-23 22:10:19 - pico-train - INFO - ├── Learning Rate: 2.95e-06
2026-10-23 22:10:19 - pico-train - INFO - └── Inf/NaN count: 0

is this as expected? what is happening + how long do I expect this to run`
```

```
just ran python -u prep_data.py !

i've done everything so far locally -- how do I connect to nautilus cluster. do I do that before doing this : PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --config_path configs/smoke.yaml

and what does this do: at runs/smoke-1/val_ppl.csv

and how do I run these 10 things overnight without keeping my laptop on?
```

```
problem with weights and biases: wandb: ERROR API key must be 40 characters long, yours was 86
```
