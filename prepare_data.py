from datasets import load_dataset, Dataset
import os
from transformers import AutoTokenizer
import json

# Load the BabyLM 2026 Strict-Small dataset
dataset = load_dataset(
    "BabyLM-community/BabyLM-2026-Strict-Small",
    split="train"
)

# Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("model")

# Checking that both loaded correctly
print(f"Number of examples: {len(dataset)}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# Testing tokenizing on one example
example = dataset[0]["text"]
tokens = tokenizer(example)

print(f"Original text: {example[:150]}")
print(f"Token IDs: {tokens['input_ids'][:20]}")

# Create reproducible training and validation splits
split_dataset = dataset.train_test_split(
    test_size=0.05,
    seed=42,
    shuffle=True
)

train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")


# Tokenize text using the GPT-2 tokenizer
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        add_special_tokens=False,
        truncation=False
    )

    # Add an end-of-text token after each example
    eos_id = tokenizer.eos_token_id

    tokenized["input_ids"] = [
        ids + [eos_id]
        for ids in tokenized["input_ids"]
    ]

    tokenized["attention_mask"] = [
        mask + [1]
        for mask in tokenized["attention_mask"]
    ]

    return tokenized


# Test the function on three examples before processing everything
sample = train_dataset.select(range(3))
tokenized_sample = tokenize_function(sample[:])

print("Tokenized examples:", len(tokenized_sample["input_ids"]))
print("First example token count:", len(tokenized_sample["input_ids"][0]))
print("Last token is EOS:", tokenized_sample["input_ids"][0][-1] == tokenizer.eos_token_id)

# Tokenize the entire training dataset
print("Tokenizing training dataset...")

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    num_proc=2
)

# Tokenize the entire validation dataset
print("Tokenizing validation dataset...")

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    num_proc=2
)

print(f"Tokenized training examples: {len(tokenized_train)}")
print(f"Tokenized validation examples: {len(tokenized_val)}")

# Number of tokens in each training sequence
BLOCK_SIZE = 256

def group_texts(examples):
    # Combine token IDs from multiple examples
    all_tokens = sum(examples["input_ids"], [])

    # Keep only complete sequences
    total_length = (len(all_tokens) // BLOCK_SIZE) * BLOCK_SIZE

    # Split into fixed-length sequences
    input_ids = [
        all_tokens[i:i + BLOCK_SIZE]
        for i in range(0, total_length, BLOCK_SIZE)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": [[1] * BLOCK_SIZE for _ in input_ids],
        "labels": [ids.copy() for ids in input_ids]
    }

# Test packing a small batch of tokenized examples
sample_batch = tokenized_train.select(range(100))
packed_sample = group_texts(sample_batch[:])

print("Number of packed sequences:", len(packed_sample["input_ids"]))

if packed_sample["input_ids"]:
    print("First sequence length:", len(packed_sample["input_ids"][0]))
    print("First 20 token IDs:", packed_sample["input_ids"][0][:20])

def pack_tokens(dataset, block_size=256):
    packed_sequences = []
    leftover = []

    for example in dataset:
        # Add this example's tokens to our leftover tokens
        leftover.extend(example["input_ids"])

        # Extract complete sequences
        while len(leftover) >= block_size:
            packed_sequences.append(leftover[:block_size])
            leftover = leftover[block_size:]

    return packed_sequences, leftover

# Test improved packing on 100 examples
test_sequences, test_leftover = pack_tokens(
    tokenized_train.select(range(100)),
    block_size=BLOCK_SIZE
)

print("Complete sequences:", len(test_sequences))
print("Tokens remaining:", len(test_leftover))
print("First sequence length:", len(test_sequences[0]))

def save_packed_dataset(dataset, output_path, block_size=256):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    leftover = []
    sequence_count = 0

    with open(output_path, "w") as f:
        for example in dataset:
            leftover.extend(example["input_ids"])

            while len(leftover) >= block_size:
                sequence = leftover[:block_size]
                leftover = leftover[block_size:]

                record = {
                    "input_ids": sequence,
                    "attention_mask": [1] * block_size,
                    "labels": sequence.copy()
                }

                f.write(json.dumps(record) + "\n")
                sequence_count += 1

    print(f"Saved {sequence_count} sequences to {output_path}")
    print(f"Final leftover tokens: {len(leftover)}")


# Pack and save training data
save_packed_dataset(
    tokenized_train,
    "data/train.jsonl",
    block_size=BLOCK_SIZE
)

# Pack and save validation data
save_packed_dataset(
    tokenized_val,
    "data/val.jsonl",
    block_size=BLOCK_SIZE
)
