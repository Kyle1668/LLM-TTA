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
        test_set = pd.read_csv("datasets/boss_benchmark/ToxicDetection/civil_comments/test.tsv", sep="\t")
        return DatasetDict(
            {
                "train": Dataset.from_pandas(train_set),
                "test": Dataset.from_pandas(test_set),
            }
        )
    else:
        return load_dataset(dataset_name)


def train_model(experiment_id, model, tokenizer, training_set, test_set):
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

    # Evalaute model
    model.eval()
    prepped_test_set = GenericDataset(test_set)
    test_loader = DataLoader(prepped_test_set, batch_size=1, shuffle=True)

    predicitons = []
    labels = []
    for eval_text, eval_label in tqdm(training_loader, desc="Evaluating Model"):
        with torch.no_grad():
            tokenized_input = tokenizer(eval_text, padding=True, truncation=True, return_tensors="pt").to(model.device)
            eval_logits = model(**tokenized_input).logits
            eval_prediciton = torch.argmax(logits, dim=1)
            predicitons.append(eval_prediciton.detach().item())
            labels.append(eval_label.detach().item())

    print(classification_report(labels, predicitons))


if __name__ == "__main__":
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    datasets = [
        # "ag_news_twitter",
        "boss_sentiment",
        "boss_toxicity",
        "boss_nli",
    ]
    base_models = [
        "bert-base-uncased",
        # "t5-large",
    ]
    num_epochs = 5

    for dataset_name in datasets:
        dataset = get_dataset(dataset_name)
        for model_name in base_models:
            # tokenizer, model = get_model_objects(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
            training_set = dataset["train"]
            test_set = dataset["test"]

            print(f"Training {model_name} on {dataset_name} for {num_epochs} epochs")
            for epoch in range(num_epochs):
                print(f"Epoch {epoch}")
                train_model(experiment_id, model, tokenizer, training_set, test_set)
