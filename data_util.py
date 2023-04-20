from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import torch
import json
import os


def get_static_examplars(dataset_name, dataset, num_shots, use_default=False):
    prompts_file_path = f"prompts/{dataset_name}_{num_shots}_shot.csv"
    if use_default:
        default_examples = json.load("prompts/defaults.json")
        # TODO: Scale with num shots
        return [default_examples["negative"][dataset_name]["prompt"]], [default_examples["positive"][dataset_name]["prompt"]]

    if os.path.exists(prompts_file_path):
        examplars = pd.read_csv(prompts_file_path, index_col=0)
        return examplars

    examplars = pd.concat([dataset[dataset["label"] == 0].sample(num_shots), dataset[dataset["label"] == 1].sample(num_shots)])
    examplars.to_csv(prompts_file_path)
    return examplars


def get_complete_prompt(input_sequence, task_instructions, exemplars, technique, edit_count=None, edit_label=None):
    # if technique == "flip":
    #     assert edit_label is not None
    #     if edit_label == 1:
    #         pos_example = input_sequence
    #     else:
    #         neg_example = input_sequence

    exemplar_strings= exemplars.apply(lambda x: f"\nSequence: {x['prompt']}\nLabel:{x['label']}", axis=1).tolist()
    formatted_input_sequence = input_sequence.replace("\n", "")
    prompt = f"""Instructions: {task_instructions}
{''.join(exemplar_strings)}

Sequence: {formatted_input_sequence}
Label:"""
    return prompt


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
    }
    hf_subset_name_mappings = {
        "toxigen": "train",
        "disaster_tweets": "train",
    }

    hf_dataset = None
    hf_path = hf_paths[set_name] if set_name in hf_paths else set_name
    if set_name in hf_subset_name_mappings:
        hf_dataset = load_dataset(hf_path, hf_subset_name_mappings[set_name], use_auth_token=True)["train"].to_pandas()
    else:
        hf_dataset = load_dataset(hf_path)["train"].to_pandas()

    formatted_dataset = pd.DataFrame()
    formatted_dataset["prompt"] = hf_dataset[hf_sets_columns_mappings[set_name][0]]
    formatted_dataset["label"] = hf_dataset[hf_sets_columns_mappings[set_name][1]]

    if sample_size:
        half_count = int(sample_size / 2)
        positives = formatted_dataset[formatted_dataset["label"] == 1].sample(half_count)
        negatives = formatted_dataset[formatted_dataset["label"] == 0].sample(sample_size - half_count)
        formatted_dataset = pd.concat([positives, negatives]).sample(frac=1).reset_index(drop=True)

    return formatted_dataset


def format_judgment(judgment_string):
    if judgment_string == "0":
        return 0
    elif judgment_string == "1":
        return 1
    else:
        return -1