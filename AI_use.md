# AI Use Documentation

## Overview

I used ChatGPT (OpenAI) throughout my BabyLM capstone project, primarily to help me understand unfamiliar tools, troubleshoot errors, and develop my Hugging Face GPT-2 training pipeline. I also used it for guidance on Nautilus, Weights & Biases (W&B), model evaluation, and GitHub repository organization.

Most of my interactions involved asking questions about how different components worked, sharing code or terminal outputs, and asking for help when something unexpectedly errored. I reviewed the suggested solutions and tested them through local experiments, Nautilus training runs, W&B logs, and evaluation results.

Below is a structured summary of the most important prompts and how I used the responses.

## 1. Understanding the Training Environment

When starting the project, I was unfamiliar with how Nautilus, Kubernetes, and my local development environment worked together.

**Prompts:**

- What exactly is  nautilus? I'm confused on where to actually write code and run the code is nautilus like vs code? Or like jupyter notebook?
- How do I set up a pod or job on kubernetes? 
- How can I make sure my job or pod is deleted? 

ChatGPT helped explain the difference between writing code locally in VS Code and running computationally intensive training jobs on Nautilus. It also provided guidance on navigating the environment and understanding how Kubernetes jobs are launched.

I followed the setup instructions, inspected terminal outputs, and eventually used Nautilus to run my training experiments.

## 2. Implementing GPT-2 with Hugging Face Trainer

For my individual implementation, I chose to train a small GPT-2 model from scratch using Hugging Face Trainer.

**Prompts:**

- How do I initialize a GPT-2 model from scratch?
- How do I set up a config file so that the hyperparameters that are swept over are the ones that I want? 
- How can I calculate perplexity using the model's evaluation loss?

ChatGPT helped me understand the Hugging Face training workflow and how the different components fit together. It also provided guidance on configuring the model and training parameters.

I implemented a GPT-2 model with 4 transformer layers, 4 attention heads, an embedding dimension of 256, and a context length of 256.

To test the implementation before running a full experiment, I performed a local 10-step training test. After confirming the config file worked, I then did the full hyperparameter sweep on Nautilus. 

I also encountered warnings related to MPS `pin_memory` and `loss_type`. ChatGPT helped me understand that these warnings were not necessarily fatal errors, and the training test completed successfully.

## 3. Dataset Preprocessing and Tokenization

I used the BabyLM 2026 Strict-Small dataset to train my model.

**Prompts:**

- How do I use the gpt2 tokenizer? 
- Does using different tokenizers affect the results? 
- How should I split the dataset into training and validation sets?

ChatGPT explained the preprocessing workflow and helped me understand how tokenization and sequence packing affect the inputs provided to the model.

I used the GPT-2 tokenizer with a vocabulary size of 50,257 and prepared the dataset using a 95% training and 5% validation split with random seed 42.

I inspected the tokenizer configuration and used the resulting preprocessed data in my training experiments.

## 4. Hyperparameter Sweeps with W&B

One of the main parts of my experiment was conducting a hyperparameter sweep to compare different training configurations.

**Prompts:**

-  Wait so the hyperparameters are the choices and the runs are basically trying to find the best possible combo? And the different collection of them affects the training time and performance?
- Where can I see the model's hyperparameters it was run on?
- How do I set up the yaml file to run the hyperparameter sweep? 

ChatGPT helped explain how sweeps select different hyperparameter combinations, how those parameters are passed into the training script, and how to interpret the resulting W&B graphs.

I configured a random sweep over learning rate (`1e-6` to `1e-4`), gradient accumulation steps (1 to 16), and learning rate scheduler (linear or cosine).

I completed 10 training runs and compared their final validation perplexities using W&B, all using Nautilus and Kubernetes. 

## 5. Running Training Jobs on Nautilus

I also used ChatGPT to help understand and troubleshoot the process of launching training jobs on Nautilus.

**Prompts:**

- How do I configure a Kubernetes job to run my training script on Nautilus?
- How can I launch W&B agents through Kubernetes jobs?
- Is there a way to request specific GPU's on nautilus? 
- My job failed in the middle. How do I resume that specific run?

ChatGPT provided guidance on Kubernetes YAML files, container environments, shared storage, and W&B agent execution.

I used Nautilus to conduct my training experiments, including a sweep configured for 10 runs with up to 3 running concurrently.

I monitored the jobs and checked W&B to verify that the experiments produced training metrics and completed runs. When a job failed (due to errors or connection problems), I had to set up another job with the hyperparameters it was using in order to continue that run (instead of starting a whole new run).

## 6. Evaluating the Trained Model

After training, I used the official BabyLM evaluation pipeline to assess the performance of my GPT-2 model.

**Prompts:**

- What is the difference between the fast and full evaluation pipelines?
- How do I run the evaluation scripts using my locally saved model?
- How do I run the eval pipeline on Nautilus? 

ChatGPT helped me understand the evaluation process, model file requirements, and how to run and document the evaluation scripts.

I ran the fast, full, and fine-tuning evaluations and saved the resulting scores as CSV files in the `results/` directory.

I verified that the evaluation scripts completed and produced benchmark scores, which I organized into CSV files for comparison and documentation.

## 7. GitHub Repository and Reproducibility

I used ChatGPT to help organize my completed implementation into a separate branch of my team's GitHub repository.

**Prompts:**

- Can you tell me the files that are not necessary to put in the shared group repo?  
- Help me write the readme so that anyone can run the eval locally
- How can I upload my trained model using Git LFS?
- How should I organize my files in my branch? 
- How can I update hardcoded paths so that the scripts work with the new repository structure?

ChatGPT provided Git commands and explanations, helped identify files needed for reproducibility, and assisted with revising the README and training script paths.

I successfully pushed the `hugging-face-gpt2-esther` branch to GitHub, including the pretrained model weights through Git LFS.

The reorganized training scripts were checked for path consistency, but I did not rerun the full training pipeline after moving them into the final repository.

## 8. Written Report Assistance

I also used ChatGPT to help brainstorm and revise portions of the checkpoint report.

**Prompts:**

- What is the difference between an abstract and an introduction?
- Can you revise this section and help with wording more academically? Give me a few suggestions

ChatGPT helped clarify research concepts, organize the introduction and literature review, and improve the wording of draft paragraphs.

I selected the relevant papers, developed the main ideas, and reviewed suggested revisions. I wrote sections of the report and asked ChatGPT to review it and give me any suggestions on how to improve it. 

## 9. Summary of AI and Human Contributions

Overall, ChatGPT served as a coding and learning assistant throughout the project. It was particularly helpful for understanding unfamiliar libraries, explaining error messages, troubleshooting training workflows, and organizing the final repository.

My contributions included choosing and configuring the GPT-2 model, preparing the dataset, executing training and hyperparameter sweeps, selecting the final checkpoint, running evaluations, and reviewing the code and documentation.