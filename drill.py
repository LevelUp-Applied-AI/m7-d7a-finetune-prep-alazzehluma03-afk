"""
Module 7 Week A — Drill: Fine-Tuning Prep.

Implement the four TODO functions. The drill does not run training — that is
tomorrow's lab. The drill exercises the mechanical preparation steps.
"""

import numpy as np
import pandas as pd
from torch import seed
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, TrainingArguments


def make_dataset(csv_path: str, test_size: float, seed: int) -> DatasetDict:
    """
    Load a CSV with `text` and `label` columns; split into train/test.

    Returns a DatasetDict with keys "train" and "test".
    """
    #  read csv_path with pandas
    df = pd.read_csv(csv_path)
    #  convert to a Hugging Face Dataset (preserve_index=False)
    raw_dataset = Dataset.from_pandas(df, preserve_index=False)    
    # split with the passed test_size and seed
    split_dataset = raw_dataset.train_test_split(test_size=test_size, seed=seed)    
    # return the resulting DatasetDict
    return split_dataset


def tokenize_dataset(ds_dict: DatasetDict, tokenizer_name: str, max_length: int) -> DatasetDict:
    """
    Tokenize all splits using the named tokenizer.

    Use truncation=True with the passed max_length. Do not pad here.
    """
    #  load tokenizer with AutoTokenizer.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # define a tokenize_fn that calls the tokenizer with truncation + max_length
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length
        )
    # apply ds_dict.map with batched=True
    tokenized_ds = ds_dict.map(tokenize_fn, batched=True)    
    #  return the tokenized DatasetDict
    return tokenized_ds


def make_training_args(output_dir: str, lr: float, epochs: int, batch_size: int, seed: int) -> TrainingArguments:
    """Build a TrainingArguments with the standard fine-tuning configuration."""
    #  return a TrainingArguments configured with the passed arguments.
    # In addition to wiring the kwargs through, set:
    #   - eval_strategy="epoch"           (renamed from evaluation_strategy in transformers 4.41+)
    #   - save_strategy="epoch"
    #   - logging_steps=50
    # The course pins transformers>=4.41,<5.0 — use the new argument names.

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_steps=50
    )
    
    args.eval_strategy = "epoch"
    args.save_strategy = "epoch"
    return args

def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.
  
    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    #  unpack eval_pred to logits, labels
    logits, labels = eval_pred
    #  argmax logits over axis 1
    predictions = np.argmax(logits, axis=1)
    #  compute accuracy and macro-F1
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    #  return as a dict
    return {
        "accuracy": acc,
        "macro_f1": f1
    }




if __name__ == "__main__":
    print("Drill 7A: import this module from tests/test_drill_7a.py to verify your implementations.")
