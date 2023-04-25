from datasets import load_dataset
from openicl import (
    DatasetReader,
    PromptTemplate,
    TopkRetriever,
    MDLRetriever,
    RandomRetriever,
    PPLInferencer,
    AccEvaluator
)
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
import os


def get_formatted_dataset(set_name, sample_size=None):
    hf_paths = {
        "toxigen": "skg/toxigen-data",
        "disaster_tweets": "venetis/disaster_tweets",
    }
    hf_sets_columns_mappings = {
        "toxigen": ("prompt", "prompt_label"),
        "disaster_tweets": ("text", "target"),
        "amazon_polarity": ("content", "label"),
        "imdb": ("text", "label"),
        "sst2": ("sentence", "label"),
        "ag_news": ("text", "label"),
    }
    hf_subset_name_mappings = {
        "toxigen": "train",
        "disaster_tweets": "train",
    }

    hf_dataset = None
    hf_path = hf_paths[set_name] if set_name in hf_paths else set_name
    if set_name in hf_subset_name_mappings:
        hf_dataset = load_dataset(hf_path, hf_subset_name_mappings[set_name], use_auth_token=True)
    else:
        hf_dataset = load_dataset(hf_path)

    if "text" not in hf_dataset["train"][0].keys():
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][0], "text")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][0], "text")
    if "label" not in hf_dataset["train"][0].keys():
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][1], "label")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][1], "label")

    return hf_dataset


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
            0: "</E>Positive Movie Review: </text>",
            1: "</E>Negative Movie Review: </text>"
        }
    elif dataset_name == "ag_news":
        tp_dict = {
            0: "</E>World (0) Article: </text>",
            1: "</E>Sports (1) Article: </text>",
            2: "</E>Business (2) Article: </text>",
            3: "</E>Sci/Tech (3) Article: </text>",
        }

    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
    return template


def main():
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_names = ["sst2", "ag_news"]
    baseline_icl_methods = ["random", "topk", "mdl"]
    model_names = [
        "distilgpt2",
        "EleutherAI/gpt-j-6b",
    ]

    for model_name in model_names:
        for dataset_name in dataset_names:
            for icl_method in baseline_icl_methods:
                print(f"Evaluating {dataset_name} with {model_name} using {icl_method}...")

                template = get_prompt_template(dataset_name)
                dataset = get_formatted_dataset(dataset_name)
                data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
                retriever = get_retriever(icl_method, data_reader)
                inferencer = PPLInferencer(model_name=model_name)

                output_file_name = f"{experiment_id}_{dataset_name}_{model_name}_{icl_method}"
                predictions = inferencer.inference(retriever, ice_template=template, output_json_filename=output_file_name)
                print(predictions)

                score = AccEvaluator().score(predictions=predictions, references=data_reader.references)
                print(f"{model_name} {dataset_name} {icl_method} Accuracy: {score}")


if __name__ == "__main__":
    main()