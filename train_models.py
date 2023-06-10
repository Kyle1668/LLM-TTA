from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset, Dataset, DatasetDict
from util_metrics import SquadMetrics
from datasets import load_dataset
from wilds import get_dataset
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os
import torch

from util_modeling import get_model_objects
from adaptive_methods import GenericDataset


def get_dataset(dataset_name):
    if dataset_name == "boss_sentiment":
        # TOOD: Add validation set and perofrm hyperparameter tuning
        train_set = pd.read_csv("datasets/boss_benchmark/SentimentAnalysis/amazon/train.tsv", sep="\t")
        train_set.rename(columns={"Text": "text", "Label": "label"}, inplace=True)
        test_set = pd.read_csv("datasets/boss_benchmark/SentimentAnalysis/amazon/test.tsv", sep="\t")
        test_set.rename(columns={"Text": "text", "Label": "label"}, inplace=True)
        return DatasetDict(
            {
                "train": Dataset.from_pandas(train_set),
                "test": Dataset.from_pandas(test_set),
            }
        )
    elif dataset_name == "boss_toxicity":
        train_set = pd.read_csv("datasets/boss_benchmark/ToxicDetection/civil_comments/train.tsv", sep="\t")
        train_set.rename(columns={"Text": "text", "Label": "label"}, inplace=True)
        test_set = pd.read_csv("datasets/boss_benchmark/ToxicDetection/civil_comments/test.tsv", sep="\t")
        test_set.rename(columns={"Text": "text", "Label": "label"}, inplace=True)
        return DatasetDict(
            {
                "train": Dataset.from_pandas(train_set),
                "test": Dataset.from_pandas(test_set),
            }
        )
    else:
        return load_dataset(dataset_name)


def train_model(model, tokenizer, training_set):
    prepped_train_set = GenericDataset(training_set)
    training_loader = DataLoader(prepped_train_set, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    batch_losses = []
    for input_batch, label_batch in tqdm(training_loader, desc="Training Model"):
        model.train()
        optimizer.zero_grad()

        tokenized_inputs = tokenizer(input_batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
        logits = model(**tokenized_inputs).logits
        loss = criterion(logits, label_batch.to(model.device))
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.detach().item())

    return sum(batch_losses) / len(batch_losses)


def evaluate_model(experiment_id, dataset_name, model, tokenizer, test_set, epoch) -> float:
    model.eval()
    prepped_test_set = GenericDataset(test_set)
    test_loader = DataLoader(prepped_test_set, batch_size=1, shuffle=True)

    predicitons = []
    labels = []
    for eval_text, eval_label in tqdm(test_loader, desc="Evaluating Model"):
        with torch.no_grad():
            tokenized_input = tokenizer(eval_text, padding=True, truncation=True, return_tensors="pt").to(model.device)
            eval_logits = model(**tokenized_input).logits
            eval_prediciton = torch.argmax(eval_logits, dim=1)
            predicitons.append(eval_prediciton.detach().item())
            labels.append(eval_label.detach().item())

    print(classification_report(labels, predicitons))
    report = classification_report(labels, predicitons, output_dict=True)

    # Save report
    if not os.path.exists("trained_models"):
        os.mkdir("trained_models")
    if not os.path.exists(f"trained_models/{experiment_id}"):
        os.mkdir(f"trained_models/{experiment_id}")
    if not os.path.exists(f"trained_models/{experiment_id}/{dataset_name}"):
        os.mkdir(f"trained_models/{experiment_id}/{dataset_name}")

    os.mkdir(f"trained_models/{experiment_id}/{dataset_name}/{model_name}_{epoch}")
    model.save_pretrained(f"trained_models/{experiment_id}/{dataset_name}/{model_name}_{epoch}")
    with open(f"trained_models/{experiment_id}/{dataset_name}/{model_name}_{epoch}/report.json", "w") as f:
        json.dump(report, f, indent=4)
    return report["accuracy"]


if __name__ == "__main__":
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    datasets = [
        {
            "name": "boss_sentiment",
            "num_classes": 3,
        },
        {
            "name": "boss_toxicity",
            "num_classes": 2,
        },
    ]
    base_models = ["bert-base-uncased"]
    num_epochs = 5

    for dataset_dict in datasets:
        dataset_name = dataset_dict["name"]
        dataset = get_dataset(dataset_name)
        for model_name in base_models:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=dataset_dict["num_classes"]).to(device)
            training_set = dataset["train"]
            test_set = dataset["test"]
            epoch_accuracies = []

            print(f"Training {model_name} on {dataset_name} for {num_epochs} epochs")
            for epoch in range(num_epochs):
                print(f"Epoch {epoch}")
                train_loss = train_model(model, tokenizer, training_set)
                test_set_perf = evaluate_model(experiment_id, dataset_name, model, tokenizer, test_set, epoch)
                epoch_accuracies.append(test_set_perf)


            # Accuracy for each epoch
            print(epoch_accuracies)
            highest_acc_epoch = np.argmax(epoch_accuracies)
            print(f"Highest accuracy (epoch {highest_acc_epoch}): {epoch_accuracies[highest_acc_epoch]}")
            # get index with largest accuracy
