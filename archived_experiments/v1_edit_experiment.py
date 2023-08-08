import os
import time
import torch
import openai_hf
import numpy as np
import pandas as pd
import plotly.express as px
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, accuracy_score


def get_gpt_inference(set_name, model_name, example_prompt, example_label, current_sequence):
    task_instructions = {
        "toxigen": "Return 1 if the following sequence is hate speech, 0 otherwise.",
        "disaster_tweets": "Return 1 if the following sequence is a disaster tweet, 0 otherwise.",
        "amazon_polarity": "Return 1 if the following sequence is a positive review, 0 otherwise."
    }
    prompt = f"""
Instructions: {task_instructions[set_name]}

Sequence: {example_prompt}
Label: {example_label}

Sequence: {current_sequence}
Label:
"""

    time.sleep(1)
    openai_api_key = os.environ["OPENAI_API_KEY"]
    try:
        response_object = openai_hf.Completion.create(model=model_name, prompt=prompt, max_tokens=10, api_key=openai_api_key)
        response_token = response_object["choices"][0]["text"]
        first_token = response_token.strip()[0]
        int_token = int(first_token)
        assert int_token in [0, 1]
        return int_token
    except:
        print(f"Error: {response_token} - unable to convert to int")
        return -1


def get_formatted_dataset(set_name, sample_size=None):
    hf_paths = {
        "toxigen": "skg/toxigen-data",
        "disaster_tweets": "venetis/disaster_tweets",
        "amazon_polarity": "amazon_polarity"
    }
    hf_sets_columns_mappings = {
        "toxigen": ("prompt", "prompt_label"),
        "disaster_tweets": ("text", "target"),
        "amazon_polarity": ("content", "label"),
    }
    hf_subset_name_mappings = {
        "toxigen": "train",
        "disaster_tweets": "train",
    }

    hf_dataset = None
    if set_name in hf_subset_name_mappings:
        hf_dataset = load_dataset(hf_paths[set_name], hf_subset_name_mappings[set_name])["train"].to_pandas()
    else:
        hf_dataset = load_dataset(hf_paths[set_name])["train"].to_pandas()

    formatted_dataset = pd.DataFrame()
    formatted_dataset["prompt"] = hf_dataset[hf_sets_columns_mappings[set_name][0]]
    formatted_dataset["label"] = hf_dataset[hf_sets_columns_mappings[set_name][1]]

    if sample_size:
        half_count = int(sample_size / 2)
        positives = formatted_dataset[formatted_dataset["label"] == 1].sample(half_count)
        negatives = formatted_dataset[formatted_dataset["label"] == 0].sample(sample_size - half_count)
        formatted_dataset = pd.concat([positives, negatives]).sample(frac=1).reset_index(drop=True)

    return formatted_dataset


def evaluate_direct_editing(experiment_id, model_name, dataset_name, dataset):
    logs = []
    default_examples = {
        "toxigen": {
            "prompt": "I hate it that muslims are now taking over the UX",
            "label": 1
        },
        "disaster_tweets": {
            "prompt": "The sky was ablaze last night in beutiful colors UX",
            "label": 0
        },
        "amazon_polarity": {
            "prompt": "I love this product UX",
            "label": 1
        }
    }

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {dataset_name} with {model_name}"):
        row = dataset.iloc[i]
        original_label = row["label"]
        current_sequence = row["prompt"]

        edit_log = {}
        edit_log["prompt"] = current_sequence
        edit_log["label"] = original_label

        default_example_prompt = default_examples[dataset_name]["prompt"]
        default_example_label = default_examples[dataset_name]["label"]
        original_judgment = get_gpt_inference(dataset_name, model_name, default_example_label, default_example_prompt, current_sequence)
        edit_log["correct_original_judgment"] = original_judgment == original_label

        # Pass in the current sequence as the prompt example with the flipped label.
        # See if doing so cuses the model to correctly output the edit label.
        flipped_original_label = 0 if original_label == 1 else 1
        flipped_judgment = get_gpt_inference(dataset_name, model_name, current_sequence, flipped_original_label, current_sequence)
        edit_log["succesful_edit"] = flipped_judgment == flipped_original_label and original_judgment != flipped_judgment

        logs.append(edit_log)

    logs_df = pd.DataFrame(logs)
    os.makedirs(f"results/{experiment_id}", exist_ok=True)
    logs_df.to_csv(f"results/{experiment_id}/{dataset_name}_{model_name}.csv", index=False)

    count_successful_direct_edits = logs_df["succesful_edit"].sum()
    edit_success_rate = 100 * count_successful_direct_edits / len(dataset)
    print(f"{dataset_name}: Edit Success - {count_successful_direct_edits}/{len(dataset)} = {edit_success_rate}%")
    print(logs_df)


def main():
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_id = f"direct_edit_experiment_{now}"
    sample_size = 500

    for dataset_name in ["disaster_tweets", "amazon_polarity", "toxigen"]:
        for model_name in ["text-davinci-003"]:
            dataset = get_formatted_dataset(dataset_name, sample_size)
            evaluate_direct_editing(experiment_id, model_name, dataset_name, dataset)

if __name__ == "__main__":
    main()
