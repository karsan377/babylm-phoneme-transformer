import math
import wandb
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset

wandb.init(project="babylm-hf-hyperparameter-sweep")

# Configure a small GPT-2 model with a 256-token context window
config = GPT2Config(
    vocab_size=50257,
    n_positions=256,
    n_ctx=256,
    n_embd=256,
    n_layer=4,
    n_head=4
)

# Initialize GPT-2 with random weights (no pretrained model)
model = GPT2LMHeadModel(config)

print("Model created successfully!")
print(f"Number of parameters: {model.num_parameters():,}")

# Load the preprocessed BabyLM training and validation datasets
datasets = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl"
    }
)

train_dataset = datasets["train"]
val_dataset = datasets["validation"]

print(f"Training sequences: {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
print(f"Tokens per sequence: {len(train_dataset[0]['input_ids'])}")

# Configure training settings
training_args = TrainingArguments(
    output_dir=f"/tmp/babylm-checkpoints/{wandb.run.id}",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=int(wandb.config.gradient_accumulation_steps),
    learning_rate=float(wandb.config.learning_rate),
    lr_scheduler_type=wandb.config.lr_scheduler_type,
    num_train_epochs=10,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    report_to="wandb",
    run_name=f"babylm-{wandb.run.id}",
)

print("Training arguments configured successfully!")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# Calculate and log perplexity whenever validation loss is reported
class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            wandb.log(
                {"eval_perplexity": perplexity},
                step=state.global_step
            )
            print(
                f"Step {state.global_step} | "
                f"Validation perplexity: {perplexity:.2f}"
            )

# Connect the GPT-2 model, datasets, and training settings
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[PerplexityCallback()]
)

print("Trainer created successfully!")

print("Starting BabyLM training...")
trainer.train()

# Save the final trained model
import os

model_output_dir = os.environ.get("MODEL_OUTPUT_DIR", "trained_models")
final_model_dir = os.path.join(model_output_dir, wandb.run.id)

trainer.save_model(final_model_dir)

print(f"Final model saved to: {final_model_dir}")

# Calculate validation perplexity after training
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])

print(f"Final validation loss: {eval_results['eval_loss']:.4f}")
print(f"Final validation perplexity: {perplexity:.2f}")

wandb.run.summary["final_eval_perplexity"] = perplexity
wandb.finish()

print("Training test completed!")
