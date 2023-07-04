from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from datasets import Dataset, DatasetDict
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import evaluate
import json
import os
import torch
import wandb

from util_modeling import get_model_objects, is_language_model, is_large_language_model
from util_data import get_formatted_dataset
from adaptive_methods import GenericDataset


def get_dataset(dataset_name, max_examples):
    local_dataset_paths = {
        "boss_sentiment": {
            "train": "datasets/boss_benchmark/SentimentAnalysis/amazon/train.tsv",
            "test": "datasets/boss_benchmark/SentimentAnalysis/amazon/test.tsv",
        },
        "boss_toxicity": {
            "train": "datasets/boss_benchmark/ToxicDetection/civil_comments/train.tsv",
            "test": "datasets/boss_benchmark/ToxicDetection/civil_comments/test.tsv",
        },
    }

    if dataset_name in local_dataset_paths:
        train_set = pd.read_csv(local_dataset_paths[dataset_name]["train"], sep="\t").dropna()
        train_set.rename(columns={"Text": "text", "Label": "label"}, inplace=True)
        if max_examples is not None:
            train_set = train_set.sample(max_examples)

        test_set = pd.read_csv(local_dataset_paths[dataset_name]["test"], sep="\t").dropna()
        test_set.rename(columns={"Text": "text", "Label": "label"}, inplace=True)
        if max_examples is not None:
            test_set = test_set.sample(max_examples)

        return DatasetDict(
            {
                "train": Dataset.from_pandas(train_set),
                "test": Dataset.from_pandas(test_set),
            }
        )

    # return load_dataset(dataset_name)
    dataset = get_formatted_dataset(dataset_name, max_examples=None)
    if dataset_name == "sst2":
        dataset["test"] = dataset["validation"]

    if max_examples is not None:
        dataset["train"] = dataset["train"].select(range(max_examples))
        dataset["test"] = dataset["test"].select(range(max_examples))

    return dataset


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, output_dict=True)
    return {
        "eval_f1": report["macro avg"]["f1-score"], "eval_acc": report["accuracy"],
        "eval_accuracy": report["accuracy"],
    }


def fine_tune_model():
    args = get_cli_args()
    num_epochs = 20
    dataset_name = args.dataset
    model_name = args.base_model

    experiment_id = f"training_{int(time())}_{args.dataset}_{args.base_model.replace('/', '_')}"
    create_exp_dir(args, experiment_id)

    wandb_run = None
    if args.use_wandb:
        project_name = "In-Context Domain Transfer Improves Out-of-Domain Robustness"
        wandb_run = wandb.init(project=project_name, group="training", name=experiment_id, config=args)

    dataset = get_dataset(dataset_name, args.max_examples)
    tokenizer, model = get_model_objects(model_name, num_labels=args.num_labels, training=True)
    tokenized_datasets = dataset.map(lambda example: tokenizer(example["text"]), batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Train the model
    training_args = TrainingArguments(
        output_dir=f"trained_models/{experiment_id}/model",
        per_device_train_batch_size=32,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=f"trained_models/{experiment_id}/logs",
        metric_for_best_model="eval_f1",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Save best model and tokenizer to its own directory
    trainer.save_model(f"trained_models/{experiment_id}/best_F1={trainer.state.best_metric}")


def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--base_model", type=str, required=False, default="bert-base-uncased")
    parser.add_argument("--max_examples", type=int, required=False, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    return args


def create_exp_dir(args, experiment_id):
    if not os.path.exists("trained_models"):
        os.mkdir("trained_models")
    os.mkdir(f"trained_models/{experiment_id}")
    json.dump(vars(args), open(f"trained_models/{experiment_id}/config.json", "w"), indent=4)


if __name__ == "__main__":
    fine_tune_model()
