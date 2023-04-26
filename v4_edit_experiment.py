from datasets import load_dataset
from openicl import (
    DatasetReader,
    PromptTemplate,
    TopkRetriever,
    MDLRetriever,
    RandomRetriever,
    PPLInferencer,
    GenInferencer
)
from datetime import datetime
from datasets import load_dataset, Dataset, DatasetDict
from accelerate import init_empty_weights
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
import os


def get_formatted_dataset(set_name, sample_size=None):
    hf_paths = {
        "sst2": "gpt3mix/sst2",
        "toxigen": "skg/toxigen-data",
        "disaster_tweets": "venetis/disaster_tweets"
    }
    hf_sets_columns_mappings = {
        "toxigen": ("prompt", "prompt_label"),
        "disaster_tweets": ("text", "target"),
        "amazon_polarity": ("content", "label"),
        "imdb": ("text", "label"),
        "sst2": ("sentence", "label"),
        "ag_news": ("text", "label"),
    }

    hf_dataset = None
    hf_path = hf_paths[set_name] if set_name in hf_paths else set_name
    hf_dataset = load_wilds_dataset(hf_path) if set_name.startswith("wilds_") else load_dataset(hf_path)

    if "text" not in hf_dataset["train"][0].keys():
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][0], "text")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][0], "text")
    if "label" not in hf_dataset["train"][0].keys():
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][1], "label")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][1], "label")

    if sample_size is not None:
        for split in ["train", "test"]:
            new_frame = None
            split_frame = hf_dataset[split].to_pandas()
            labels = split_frame["label"].unique()
            sample_size_per_label = sample_size // len(labels)
            for label in labels:
                label_samples = split_frame[split_frame["label"] == label].sample(sample_size_per_label)
                if new_frame is None:
                    new_frame = label_samples
                else:
                    new_frame = pd.concat([new_frame, label_samples])

            new_frame = new_frame.sample(frac=1)
            hf_dataset[split] = Dataset.from_pandas(new_frame)

    return hf_dataset


def load_wilds_dataset(dataset_name):
    if dataset_name == "wilds_civil_comments":
        dataset = get_dataset(dataset="civilcomments", download=True)
        train_dict = {
            "text": [],
            "label": [],
            "group": []
        }
        for text, label, group in dataset.get_subset("train"):
            train_dict["text"].append(text)
            train_dict["label"].append(label.item())
            train_dict["group"].append(group.tolist())

        test_dict = {
            "text": [],
            "label": [],
            "group": []
        }
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
        train_dict = {
            "text": [],
            "label": [],
            "group": []
        }
        for content, label, group in dataset.get_subset("train"):
            train_dict["text"].append(content)
            train_dict["label"].append(label.item())
            train_dict["group"].append(group.tolist())

        test_dict = {
            "text": [],
            "label": [],
            "group": []
        }
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


def get_retriever(icl_method, data, ice_num=8, index_split='train', test_split='test'):
    if icl_method == "topk":
        return TopkRetriever(dataset_reader=data, ice_num=ice_num, index_split=index_split, test_split=test_split)
    elif icl_method == "mdl":
        return MDLRetriever(dataset_reader=data, ice_num=ice_num, index_split=index_split, test_split=test_split)
    elif icl_method == "random":
        return RandomRetriever(dataset_reader=data, ice_num=ice_num, index_split=index_split, test_split=test_split)
    else:
        raise Exception("Invalid ICL method")


def get_prompt_template(dataset_name):
    tp_dict = None
    if dataset_name == "sst2":
        tp_dict = {
            0: "</E>Negative Movie Review: </text>",
            1: "</E>Positive Movie Review: </text>"
        }
    elif dataset_name == "ag_news":
        tp_dict = {
            0: "</E>World (0) Article: </text>",
            1: "</E>Sports (1) Article: </text>",
            2: "</E>Business (2) Article: </text>",
            3: "</E>Sci/Tech (3) Article: </text>",
        }
    elif dataset_name == "toxigen":
        tp_dict = {
            0: "</E>Non-Toxic: </text>",
            1: "</E>Toxic: </text>"
        }
    elif dataset_name == "disaster_tweets":
        tp_dict = {
            0: "</E>Non-Disaster Tweet: </text>",
            1: "</E>Disaster Tweet: </text>"
        }
    elif dataset_name == "wilds_civil_comments":
        tp_dict = {
            0: "</E>Not Hate Speech: </text>",
            1: "</E>Hate Speech: </text>"
        }
    elif dataset_name == "wilds_amazon":
        tp_dict = {
            0: "</E>1 Start Review: </text>",
            1: "</E>2 Start Review: </text>",
            2: "</E>3 Start Review: </text>",
            3: "</E>4 Start Review: </text>",
            4: "</E>5 Start Review: </text>",
        }

    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
    return template


def main():
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_names = ["wilds_civil_comments", "ag_news", "wilds_amazon"]
    baseline_icl_methods = ["topk", "random"]
    model_names = [
        # "decapoda-research/llama-7b-hf",
        # "EleutherAI/pythia-2.8b",
        # "EleutherAI/pythia-1b",
        "EleutherAI/pythia-410m"
    ]
    reports = []

    for model_name in model_names:
        print(f"Loading model {model_name}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        for dataset_name in dataset_names:
            print(f"Loading dataset {dataset_name}...")
            dataset = get_formatted_dataset(dataset_name, sample_size=5000)
            for icl_method in baseline_icl_methods:
                print(f"Evaluating {dataset_name} with {model_name} using {icl_method}...")

                template = get_prompt_template(dataset_name)
                data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
                retriever = get_retriever(icl_method, data_reader)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                inferencer = PPLInferencer(model=model)
                formatted_path_name = dataset_name.replace("_", "-")
                output_file_name = f"{experiment_id}_{dataset_name}_{formatted_path_name}_{icl_method}"
                predictions = inferencer.inference(retriever, ice_template=template, output_json_filename=output_file_name)

                if not os.path.exists(f"results/{experiment_id}"):
                    os.makedirs(f"results/{experiment_id}")

                report_dict = classification_report(data_reader.references, predictions, output_dict=True)
                json.dump(report_dict, open(f"results/{experiment_id}/{output_file_name}_report.json", "w+"), indent=4)
                print(f"Classification Results: {formatted_path_name} {dataset_name} {icl_method}")
                print(classification_report(data_reader.references, predictions))
                reports.append({
                    "dataset": dataset_name,
                    "icl_method": icl_method,
                    "model": formatted_path_name,
                    "accuracy": report_dict["accuracy"],
                    "macro avg f1-score": report_dict["macro avg"]["f1-score"],
                    "macro avg precision": report_dict["macro avg"]["precision"],
                    "macro avg recall": report_dict["macro avg"]["recall"],
                })

    all_reports = pd.DataFrame(reports)
    print(all_reports)
    all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)



if __name__ == "__main__":
    main()