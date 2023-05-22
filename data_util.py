from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from faiss import IndexIDMap, IndexFlatIP
from datasets import load_dataset
from datetime import datetime
from wilds import get_dataset
from openicl import DatasetReader, PromptTemplate, TopkRetriever, MDLRetriever, RandomRetriever, PPLInferencer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
import os


def generate_icl_report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, num_successfull_edits):
    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    formatted_model_name = model_name.replace("/", "-")
    report_dict = classification_report([entry["label"] for entry in dataset[eval_set.replace("+adaptive", "")]], original_judgments, output_dict=True)
    num_edits = len(dataset["edits"]) if "edits" in dataset else -1
    edit_success_rate = num_successfull_edits / num_edits if num_edits > 0 else -1
    icl_report = {
        "dataset": dataset_name,
        "split": eval_set,
        "dataset size": len(dataset[eval_set.replace("+adaptive", "")]),
        "icl_method": icl_method,
        "model": formatted_model_name,
        "accuracy": report_dict["accuracy"],
        "avg precision": report_dict["macro avg"]["precision"],
        "avg recall": report_dict["macro avg"]["recall"],
        "avg f1": report_dict["macro avg"]["f1-score"],
        "num edits": num_edits,
        "num successfull edits": num_successfull_edits,
        "edit success rate": edit_success_rate,
    }
    output_file_name = f"set={dataset_name}_split={eval_set}_method={icl_method}_model={formatted_model_name}"

    if eval_set == "prod":
        json.dump(icl_report, open(f"results/{experiment_id}/{output_file_name}_report.json", "w+"), indent=4)
        print(f"Classification Results: {formatted_model_name} {dataset_name} {icl_method}")
        print(classification_report(data_reader.references, original_judgments))
        confusion_matrix_fig = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(data_reader.references, original_judgments))
        confusion_matrix_fig.figure_.savefig(f"results/{experiment_id}/{output_file_name}_confusion_matrix.png")

    return icl_report


def get_formatted_dataset(set_name, max_examples=None):
    hf_paths = {"sst2": "gpt3mix/sst2", "toxigen": "skg/toxigen-data", "disaster_tweets": "venetis/disaster_tweets"}
    hf_sets_columns_mappings = {
        "toxigen": ("prompt", "prompt_label"),
        "disaster_tweets": ("text", "target"),
        "amazon_polarity": ("content", "label"),
        "imdb": ("text", "label"),
        "sst2": ("sentence", "label"),
        "ag_news": ("text", "label"),
        "squad": ("context", "answers", "question"),
    }

    hf_dataset = None
    hf_path = hf_paths[set_name] if set_name in hf_paths else set_name
    if set_name.startswith("wilds_"):
        hf_dataset = load_wilds_dataset(hf_path)
    elif set_name == "scotus":
        hf_dataset = load_scotus_dataset()
    elif set_name == "ag_news":
        hf_dataset = load_shifted_agnews_dataset()
    elif set_name == "civil_toxigen":
        hf_dataset = load_civil_comments_and_toxigen_dataset()
    elif set_name == "rotten_tomatoes_imdb":
        hf_dataset = DatasetDict({"train": load_dataset("rotten_tomatoes", split="train"), "test": load_dataset("imdb", split="test")})
    else:
        hf_dataset = load_dataset(hf_path)

    if "text" not in hf_dataset["train"][0].keys():
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][0], "text")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][0], "text")
    if "label" not in hf_dataset["train"][0].keys():
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][1], "label")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][1], "label")
    if is_qa_task := set_name == "squad":
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][2], "question")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][2], "question")

    # Create a validation set from the same distirbution as the train set - if none already exist
    if "validation" not in hf_dataset.keys():
        train_set = hf_dataset["train"].to_pandas()
        validation_set = train_set.sample(frac=0.2)
        train_set = train_set.drop(validation_set.index)
        hf_dataset["train"] = Dataset.from_pandas(train_set)
        hf_dataset["validation"] = Dataset.from_pandas(validation_set)

    if max_examples is not None:
        for split in ["train", "validation", "test"]:
            if max_examples >= len(hf_dataset[split]):
                print(f"WARNING: max_examples ({max_examples}) is greater than the number of examples in the {split} set ({len(hf_dataset[split])}).")
                continue

            new_frame = None
            if is_qa_task:
                new_frame.sample(max_examples)
            else:
                split_frame = hf_dataset[split].to_pandas()
                labels = split_frame["label"].unique()
                max_examples_per_label = max_examples // len(labels)
                for label in labels:
                    current_label_sample_size = max_examples_per_label if len(split_frame[split_frame["label"] == label]) > max_examples_per_label else len(split_frame[split_frame["label"] == label])
                    label_samples = split_frame[split_frame["label"] == label].sample(current_label_sample_size)
                    if new_frame is None:
                        new_frame = label_samples
                    else:
                        new_frame = pd.concat([new_frame, label_samples])

            new_frame = new_frame.sample(frac=1)
            new_frame = new_frame.drop(columns=["__index_level_0__"]) if "__index_level_0__" in new_frame.columns else new_frame
            hf_dataset[split] = Dataset.from_pandas(new_frame)

    # Split the test set into a production traffic set from which edits will be made, and a holdout set
    if enable_edits := False:
        original_test_set = hf_dataset["test"].to_pandas().drop(columns=["__index_level_0__"])
        edit_set = original_test_set.sample(frac=0.5)
        test_set = original_test_set.drop(edit_set.index)
        hf_dataset["test"] = Dataset.from_pandas(test_set)
        hf_dataset["prod"] = Dataset.from_pandas(edit_set)

    return hf_dataset


def load_civil_comments_and_toxigen_dataset() -> DatasetDict:
    civil_comments = load_wilds_dataset("wilds_civil_comments")
    toxigen = load_dataset("skg/toxigen-data", "train", use_auth_token=True).rename_column("generation", "text").rename_column("prompt_label", "label")
    formatted_toxigen = toxigen["train"].map(lambda x: {"text": x["text"].replace("- ", "").split("\\n")[0]})
    return DatasetDict(
        {
            "train": formatted_toxigen,
            "test": civil_comments["test"],
        }
    )


def load_scotus_dataset():
    train_set = pd.read_csv("datasets/scotus_train.csv")
    test_set = pd.read_csv("datasets/scotus_test.csv")
    full_dataset = DatasetDict()
    full_dataset["train"] = Dataset.from_pandas(train_set)
    full_dataset["test"] = Dataset.from_pandas(test_set)
    return full_dataset


def load_shifted_agnews_dataset():
    full_dataset = DatasetDict()
    full_dataset["train"] = Dataset.from_pandas(pd.read_csv("datasets/ag_news_train.csv"))
    full_dataset["test"] = load_dataset("ag_news", split="test")
    return full_dataset


def load_wilds_dataset(dataset_name):
    if dataset_name == "wilds_civil_comments":
        dataset = get_dataset(dataset="civilcomments", download=True)
        train_dict = {"text": [], "label": [], "group": []}
        for text, label, group in dataset.get_subset("train"):
            train_dict["text"].append(text)
            train_dict["label"].append(label.item())
            train_dict["group"].append(group.tolist())

        test_dict = {"text": [], "label": [], "group": []}
        for text, label, group in dataset.get_subset("test"):
            test_dict["text"].append(text)
            test_dict["label"].append(label.item())
            test_dict["group"].append(group.tolist())

        full_dataset = DatasetDict()
        full_dataset["train"] = Dataset.from_pandas(pd.DataFrame(train_dict))
        full_dataset["test"] = Dataset.from_pandas(pd.DataFrame(test_dict))
        return full_dataset
    elif dataset_name == "wilds_amazon":
        dataset = get_dataset(dataset="amazon", download=True)
        train_dict = {"text": [], "label": [], "group": []}
        for content, label, group in dataset.get_subset("train"):
            train_dict["text"].append(content)
            train_dict["label"].append(label.item())
            train_dict["group"].append(group.tolist())

        test_dict = {"text": [], "label": [], "group": []}
        for content, label, group in dataset.get_subset("test"):
            test_dict["text"].append(content)
            test_dict["label"].append(label.item())
            test_dict["group"].append(group.tolist())

        full_dataset = DatasetDict()
        full_dataset["train"] = Dataset.from_pandas(pd.DataFrame(train_dict))
        full_dataset["test"] = Dataset.from_pandas(pd.DataFrame(test_dict))
        return full_dataset
    else:
        raise Exception("Invalid WILDS dataset")
