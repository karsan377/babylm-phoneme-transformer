# AI Use

Summarized prompts I used:


## Repository setup

- Asked for help restructuring the local folder: a nested clone, an empty outer repo, and a leftover `upstream` remote. Claude moved the files up and fixed the remote names
- 

## Repository cleanup decisions

- Asked which files to track. Claude removed the `.gitmodules` entries, which pointed at folders that weren't real submodules, and added `evaluation_data/` to `.gitignore`
- Chose to keep `data/phoible.csv` in the repo

## Fine-tuning dependency

- Asked why fine-tuning wasn't working. Claude found that `scripts/run_glue_evaluation.py` calls `finetune_classification.py` from a submodule that was empty in the snapshot
- Asked Claude to add the submodule back on a feature branch. Claude added it and pushed the branch. I have not yet tested that the script runs

## Tokenizer access

- Asked where the phoneme tokenizers live. The configs load them from Hugging Face by name. The repos returned 401 and the organization returned 404 when checked anonymously
- Asked whether the tokenizers still exist and whether I could rebuild one. Claude confirmed nothing was cached locally. The question is with my mentor
- Asked for help wording a message to my mentor and for the question to be clearer. Claude suggested versions that named the exact repos and config files

## Code and configuration written with Claude
- Claude wrote and edited the training and data code in this repo: `train.py`, `src/preprocessing.py`, and `src/config.py`, along with the dataset config for BabyLM Strict-Small phonemes
- Claude edited the tokenizer configs and `requirements.txt`, and updated the README. I reviewed the changes and asked for revisions where needed
- I decided what to run, which model size to use, and which results to report. Claude did not choose the experiment settings.

## Pretraining sweep

- used Claude to write Kube job specs for the cluster and intermittently monitor them. It ran smoke tests and then the full hyperparameter sweep on my behalf
- used it to check job status, clean up stale jobs, export W&B results, and plot results


## Other questions

- Asked for a handoff summary to paste into a new session.
- Asked for explanations of course concepts and how they related to the process and results 
- Asked Claude to rewrite the README so someone else could reproduce the runs. I asked for no separate markdown file




